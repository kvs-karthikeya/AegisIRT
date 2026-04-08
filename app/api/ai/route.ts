import { NextResponse, type NextRequest } from "next/server";
import { GoogleGenAI } from "@google/genai";
import fs from "fs";
import path from "path";
import puppeteer, { Browser } from "puppeteer";

// Global browser cache to speed up subsequent requests
let globalBrowser: Browser | null = null;
const snapshotPath = path.join(process.cwd(), "snapshot.json");
const healingLogPath = path.join(process.cwd(), "healing_log.json");

// Ensure files exist (wrapped in try/catch to survive Vercel's Read-Only Serverless FS)
try {
  if (!fs.existsSync(snapshotPath)) {
    fs.writeFileSync(snapshotPath, JSON.stringify({ html: "<div id='app'>\\n  <button id='v1_enroll_btn'>Enroll Patient</button>\\n</div>" }, null, 2));
  }
  if (!fs.existsSync(healingLogPath)) {
    fs.writeFileSync(healingLogPath, JSON.stringify([], null, 2));
  }
} catch (e) {
  console.warn("Skipping initial file write: environment is likely a read-only Vercel instance.");
}

function domDiff(snapshotHtml: string, currentHtml: string) {
  const diffs: string[] = [];
  try {
    const snapMatches = snapshotHtml.match(/id=['"](.*?)['"]/g) || [];
    const curMatches = currentHtml.match(/id=['"](.*?)['"]/g) || [];
    
    const snapshotIds = snapMatches.map(s => s.replace(/id=|['"]/g, ''));
    const currentIds = curMatches.map(s => s.replace(/id=|['"]/g, ''));

    snapshotIds.forEach(id => {
      if (!currentIds.includes(id)) {
        diffs.push(`ID '${id}' was removed or renamed.`);
      }
    });
  } catch(e) { /* ignore parse errors */ }

  return diffs.length > 0 ? diffs.join(" ") : "No structural ID changes detected.";
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    // Phase 4: Feedback logging system
    if (action === "provide_feedback") {
      const { actual_success, result } = body;
      let logs: any[] = [];
      try {
        logs = JSON.parse(fs.readFileSync(healingLogPath, "utf-8"));
      } catch (e) { /* default empty logs */ }
      
      if (result) {
        logs.push({
          timestamp: new Date().toISOString(),
          broken_locator: result.original_locator,
          healed_locator: result.ai_suggestion,
          html_context: result.clinical_analysis?.context || "general_irt", 
          actual_success,
          user_feedback: actual_success ? "approve" : "reject"
        });
        
        try {
          fs.writeFileSync(healingLogPath, JSON.stringify(logs, null, 2));
        } catch (e) { }
      }
      return NextResponse.json({ status: "success", message: "Feedback logged" });
    }

    if (action === "get_stats") {
      return NextResponse.json({ server_status: "online", model_loaded: true });
    }

    if (action === "heal_batch") {
      const { test_code, html_content, clinical_context } = body;

      // Ensure snapshot exists or parse fallback
      let snapshot = { html: "" };
      let initialSnapshotSaved = false;
      try { 
        const fileContent = fs.readFileSync(snapshotPath, "utf-8");
        snapshot = JSON.parse(fileContent); 
        if (!snapshot.html) throw new Error("Empty snapshot");
      } catch(e) { 
        snapshot = { html: html_content }; 
        try { fs.writeFileSync(snapshotPath, JSON.stringify(snapshot, null, 2)); } catch (err) {}
        initialSnapshotSaved = true;
      }
      const diffResult = initialSnapshotSaved ? "Initial snapshot saved, no diff available." : domDiff(snapshot.html, html_content);

      // Extract locators
      const keywords = ['locator', 'getByTestId', 'getByText', 'getByRole', 'getByLabel', 'findElement', 'find_element', 'FindElement'];
      const locators: { snippet: string, keyword: string }[] = [];
      
      for (const keyword of keywords) {
        let index = 0;
        while ((index = test_code.indexOf(keyword, index)) !== -1) {
          let openParenIdx = test_code.indexOf('(', index);
          if (openParenIdx !== -1 && openParenIdx < index + keyword.length + 2) {
            let parenCount = 1;
            let curr = openParenIdx + 1;
            let inString = false;
            let stringChar = '';
            while (curr < test_code.length && parenCount > 0) {
              const char = test_code[curr];
              if (inString) {
                 if (char === stringChar && test_code[curr-1] !== '\\') inString = false;
              } else {
                 if (char === "'" || char === '"' || char === '`') { inString = true; stringChar = char; }
                 else if (char === '(') parenCount++;
                 else if (char === ')') parenCount--;
              }
              curr++;
            }
            if (parenCount === 0) {
              const snippet = test_code.substring(index, curr);
              locators.push({ snippet, keyword });
            }
          }
          index += keyword.length;
        }
      }

      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "dummy" });
      let browser = globalBrowser;
      try {
        if (!browser || !browser.connected) {
          browser = await puppeteer.launch({ headless: true });
          globalBrowser = browser;
        }
      } catch (e) {
        console.warn("Puppeteer failed to launch", e);
      }

      let healed_code = test_code;
      const batchResults = [];

      const debugLogPath = path.join(process.cwd(), "healing_debug.log");
      fs.writeFileSync(debugLogPath, "--- NEW BATCH HEALING START ---\n");
      const debugLog = (msg: string) => {
          fs.appendFileSync(debugLogPath, msg + "\n");
          console.log(msg);
      };

      // Ensure locator representation is concise
      const locatorMap = locators.map((loc, i) => ({ 
        id: i, 
        snippet: loc.snippet 
      }));
      
      const aiPrompt = `You are a test automation healing engine working on a clinical trial IRT system. Your job is to analyze the HTML and find the new elements matching a list of broken locators.\n\nDOM Diff Info:\n${diffResult}\n\nRules: never return placeholder or mock values, do not invent selectors. Verify your finding against the HTML. If you confidently cannot find a selector, return a value of "null".\n\nBroken Locators List:\n${JSON.stringify(locatorMap, null, 2)}\n\nHTML Code:\n${html_content}\n\nYou must return a JSON array of objects. Each object must have these exact fields: id (matching the broken locator id integer), type (CSS_SELECTOR, XPATH, ID, NAME, CLASS_NAME), value (the found string selector), confidence (number 0-1), reasoning (string). Output ONLY the raw JSON array, without any markdown formatting like \`\`\`json.`;

      debugLog(`[Batch Processing] Sending ${locators.length} locators in a single request to Gemini. HTML length: ${html_content.length}`);
      
      let parsedAiChoices: any[] = [];
      if (locators.length > 0) {
        try {
          const response = await ai.models.generateContent({
              model: "gemini-2.5-flash",
              contents: aiPrompt,
              config: { 
                  responseMimeType: "application/json", 
                  temperature: 0.1,
              }
          });
          const responseText = response.text || "[]";
          debugLog(`[Raw Gemini Response length]: ${responseText.length} chars`);
          const match = responseText.match(/\[[\s\S]*\]/);
          parsedAiChoices = JSON.parse(match ? match[0] : responseText);
          debugLog(`[Parsed Target Array Length]: ${parsedAiChoices.length}`);
        } catch(e: any) {
          debugLog(`[Gemini API Error or parse fail]: ${e.message}`);
        }
      }

      let page: any = null;
      if (browser) {
          page = await browser.newPage();
          await page.setContent(html_content, { waitUntil: 'load' });
      }

      for (let i = 0; i < locators.length; i++) {
        const loc = locators[i];
        const choice = parsedAiChoices.find((c: any) => c.id === i);
        
        let finalLocator = { type: "", value: "", replacement_code: loc.snippet, confidence: 0 };
        let verificationResult = false;
        let screenshotBase64 = "";

        if (choice && choice.value && choice.value !== "null") {
            let t = choice.type;
            let v = choice.value;
            let quote = v.includes("'") ? '"' : "'";
            let newLocatorCode = "";
            if (t === "ID") newLocatorCode = `find_element(By.ID, ${quote}${v}${quote})`;
            else if (t === "CSS_SELECTOR") newLocatorCode = `find_element(By.CSS_SELECTOR, ${quote}${v}${quote})`;
            else if (t === "XPATH") newLocatorCode = `find_element(By.XPATH, ${quote}${v}${quote})`;
            else if (t === "NAME") newLocatorCode = `find_element(By.NAME, ${quote}${v}${quote})`;
            else if (t === "CLASS_NAME") newLocatorCode = `find_element(By.CLASS_NAME, ${quote}${v}${quote})`;
            else newLocatorCode = `find_element(By.${t}, ${quote}${v}${quote})`;

            finalLocator = { 
              type: t, 
              value: v, 
              replacement_code: newLocatorCode,
              confidence: parseFloat(choice.confidence) || 0.9 
            };

            if (page) {
               let qsSelector = v;
               if (t === "ID") qsSelector = "#" + v;
               else if (t === "NAME") qsSelector = "[name='" + v + "']";
               else if (t === "CLASS_NAME") qsSelector = "." + v;

               const isXPath = t.includes("XPATH") || v.startsWith("//");
               
               const isFound = await page.evaluate((locValue: string, isXPathArg: boolean, querySel: string) => {
                 try {
                   let el;
                   if (isXPathArg) {
                     const docRes = document.evaluate(locValue, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                     el = docRes.singleNodeValue;
                   } else {
                     el = document.querySelector(querySel);
                   }
                   if (el) {
                     (el as HTMLElement).style.border = "4px solid #ffffff";
                     (el as HTMLElement).style.backgroundColor = "#1e1e1e";
                     (el as HTMLElement).style.color = "#ffffff";
                     return true;
                   }
                 } catch (e) { }
                 return false;
               }, v, isXPath, qsSelector);

               debugLog(`[Puppeteer Verification]: ${t}='${v}' => ${isFound}`);

               if (isFound) {
                 verificationResult = true;
                 const buffer = await page.screenshot({ encoding: "base64" });
                 screenshotBase64 = buffer as string;
                 
                 // Clean up highlight
                 await page.evaluate((locValue: string, isXPathArg: boolean, querySel: string) => {
                     try {
                       let el;
                       if (isXPathArg) {
                         const docRes = document.evaluate(locValue, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                         el = docRes.singleNodeValue;
                       } else {
                         el = document.querySelector(querySel);
                       }
                       if (el) {
                         (el as HTMLElement).style.border = "none";
                         (el as HTMLElement).style.backgroundColor = "transparent";
                         (el as HTMLElement).style.color = "inherit";
                       }
                     } catch (e) { }
                 }, v, isXPath, qsSelector);
               } else {
                 debugLog(`[Puppeteer Verification]: FAILED/NULL for '${v}'`);
               }
            }
        } else {
            debugLog(`[Skipped Verification]: AI did not suggest a valid choice for '${loc.snippet}'`);
        }

        if (verificationResult) {
          healed_code = healed_code.replace(loc.snippet, finalLocator.replacement_code);
        } else {
          const snippetIndex = healed_code.indexOf(loc.snippet);
          if (snippetIndex !== -1) {
              let lineStart = healed_code.lastIndexOf('\n', snippetIndex);
              lineStart = lineStart === -1 ? 0 : lineStart + 1;
              const indentMatch = healed_code.substring(lineStart, snippetIndex).match(/^\s*/);
              const indent = indentMatch ? indentMatch[0] : "";
              
              let locatorArgs = loc.snippet;
              const argsMatch = loc.snippet.match(/find_element\((.*?)\)/);
              if (argsMatch && argsMatch[1]) locatorArgs = argsMatch[1];
              
              healed_code = healed_code.substring(0, lineStart) + 
                            `${indent}# AEGISIRT COULD NOT HEAL THIS LOCATOR: ${locatorArgs}\n` + 
                            healed_code.substring(lineStart);
          }
        }

        batchResults.push({
          id: `healing_${Date.now()}_${Math.random().toString(36).substring(7)}`,
          element_name: loc.snippet,
          original_locator: { type: loc.keyword, value: loc.snippet },
          ai_suggestion: finalLocator,
          reasoning: choice?.reasoning || "Failed to identify new locator or LLM missed field.",
          status: verificationResult ? "verified" : "unverified",
          verifiedResult: verificationResult,
          screenshot: screenshotBase64,
          model_info: { language: "auto" }
        });
      }

      if (page) await page.close();

      return NextResponse.json({
        healed_test_code: healed_code,
        results: batchResults
      });
    }
    
    return NextResponse.json({ error: "Unknown action" }, { status: 400 });
  } catch (error: any) {
    console.error("AI Error:", error);
    return NextResponse.json({ error: "AegisIRT AI failed", details: error.message }, { status: 500 });
  }
}
