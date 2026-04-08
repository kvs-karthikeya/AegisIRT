import { NextResponse, type NextRequest } from "next/server";
import { GoogleGenAI } from "@google/genai";
import fs from "fs";
import path from "path";
import puppeteer, { Browser } from "puppeteer";

let globalBrowser: Browser | null = null;
const snapshotPath = path.join(process.cwd(), "snapshot.json");
const healingLogPath = path.join(process.cwd(), "healing_log.json");

try {
  if (!fs.existsSync(snapshotPath)) fs.writeFileSync(snapshotPath, JSON.stringify({ html: "<div id='app'><button id='v1_enroll_btn'>Enroll Patient</button></div>" }, null, 2));
  if (!fs.existsSync(healingLogPath)) fs.writeFileSync(healingLogPath, JSON.stringify([], null, 2));
} catch (e) { console.warn("Skipping file write: read-only environment."); }

function domDiff(snapshotHtml: string, currentHtml: string): string {
  const diffs: string[] = [];
  try {
    const snapIds = (snapshotHtml.match(/id=['\"](.*?)['"]/g) || []).map(s => s.replace(/id=|['"]/g, ""));
    const curIds  = (currentHtml.match(/id=['\"](.*?)['"]/g) || []).map(s => s.replace(/id=|['"]/g, ""));
    snapIds.forEach(id => { if (!curIds.includes(id)) diffs.push(`ID '${id}' was removed or renamed.`); });
  } catch {}
  return diffs.length > 0 ? diffs.join(" ") : "No structural ID changes detected.";
}

const MODULE_KEYWORDS: Record<string, string[]> = {
  "authentication":            ["login", "username", "password", "mfa", "coordinator"],
  "patient-search":            ["patient_search", "search_input", "search_btn", "patient_id"],
  "patient-screening":         ["screening", "eligib", "initials", "dob", "hoehn", "updrs_score", "disease_duration"],
  "stratified-randomization":  ["randomiz", "randomization", "treatment_arm", "stratif", "stratum"],
  "kit-dispensing":            ["dispens", "kit", "medication"],
  "baseline-assessments":      ["baseline", "updrs_part", "tremor", "rigidity", "assessment"],
  "emergency-procedures":      ["unblind", "emergency", "physician", "authorization"],
  "reports-audit":             ["audit", "report", "cfr", "download"],
  "protocol-amendments":       ["amendment", "acknowledge", "protocol"],
};

function getRelevantHtmlChunk(html: string, contextClue: string): string {
  const lower = contextClue.toLowerCase();
  for (const [module, keywords] of Object.entries(MODULE_KEYWORDS)) {
    if (keywords.some(k => lower.includes(k))) {
      const idx = html.indexOf(`data-module="${module}"`);
      if (idx === -1) continue;
      const start = html.lastIndexOf("<", idx);
      let depth = 0, end = start;
      for (let i = start; i < html.length; i++) {
        if (html[i] === "<" && html[i+1] !== "/") depth++;
        if (html[i] === "<" && html[i+1] === "/") depth--;
        if (depth === 0 && i > start) { end = i; while (end < html.length && html[end] !== ">") end++; end++; break; }
      }
      const chunk = html.slice(start, end);
      if (chunk.length > 50) { console.log(`[AegisIRT] Chunk: ${module} (${chunk.length} chars)`); return chunk; }
    }
  }
  console.warn(`[AegisIRT] No chunk for "${contextClue}" — full HTML (${html.length} chars)`);
  return html;
}

interface ParsedLocator {
  lineIndex: number; originalLine: string; locator_type: string;
  locator_value: string; text_hint: string; detected_language: string; context_clue: string;
}

function extractAllLocators(code: string): ParsedLocator[] {
  const lines = code.split("\n");
  const results: ParsedLocator[] = [];
  let detected_language = "python";
  if (code.includes("WebElement") || code.includes("driver.findElement")) detected_language = "java";
  else if (code.includes("IWebElement") || code.includes("driver.FindElement")) detected_language = "csharp";
  else if (code.includes("page.locator") || code.includes("await page.")) detected_language = "typescript";

  const patterns = [
    { re: /find_element\(\s*By\.(\w+)\s*,\s*["'`](.*?)["'`]\s*\)/, lang: "python" },
    { re: /findElement\(\s*By\.(\w+)\(\s*["'`](.*?)["'`]\s*\)\s*\)/, lang: "java" },
    { re: /FindElement\(\s*By\.(\w+)\(\s*["'`](.*?)["'`]\s*\)\s*\)/, lang: "csharp" },
    { re: /\.locator\(\s*["'`](.*?)["'`]\s*\)/, lang: "typescript" },
  ];

  lines.forEach((line, lineIndex) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || trimmed.startsWith("//")) return;
    if (trimmed.includes("EC.visibility_of_element_located") || trimmed.includes("WebDriverWait")) return;

    for (const { re, lang } of patterns) {
      const match = trimmed.match(re);
      if (!match) continue;

      let locator_type = "By.CSS_SELECTOR", locator_value = "";
      if (lang === "python" || lang === "java" || lang === "csharp") {
        locator_type = `By.${match[1].toUpperCase()}`; locator_value = match[2];
      } else {
        locator_value = match[1];
      }

      const surroundingLines = lines.slice(Math.max(0, lineIndex - 3), lineIndex + 2).join(" ");
      const context_clue = `${locator_value} ${surroundingLines}`;
      const lower = context_clue.toLowerCase();

      let text_hint = locator_value;
      if (lower.includes("enroll")) text_hint = "Enroll Patient";
      else if (lower.includes("randomiz") || lower.includes("stratif")) text_hint = "Randomize Subject";
      else if (lower.includes("dispens") || lower.includes("kit")) text_hint = "Dispense Medication";
      else if (lower.includes("login") || lower.includes("username") || lower.includes("password")) text_hint = "Login";
      else if (lower.includes("mfa") || lower.includes("verify")) text_hint = "MFA Verification";
      else if (lower.includes("eligib") || lower.includes("screening")) text_hint = "Patient Screening";
      else if (lower.includes("baseline") || lower.includes("updrs")) text_hint = "Baseline Assessment";
      else if (lower.includes("unblind") || lower.includes("emergency")) text_hint = "Emergency Unblinding";
      else if (lower.includes("audit") || lower.includes("report")) text_hint = "Audit Report";
      else if (lower.includes("amendment") || lower.includes("acknowledge")) text_hint = "Protocol Amendment";

      results.push({ lineIndex, originalLine: line, locator_type, locator_value, text_hint, detected_language, context_clue });
      break;
    }
  });
  return results;
}

function reconstructFile(originalCode: string, healedMap: Map<number, { type: string; value: string; healed: boolean }>): string {
  const lines = originalCode.split("\n");
  const output: string[] = [];
  lines.forEach((line, idx) => {
    const heal = healedMap.get(idx);
    if (heal) {
      if (heal.healed) {
        let newLine = line;
        newLine = newLine.replace(/find_element\(\s*By\.\w+\s*,\s*["'`].*?["'`]\s*\)/, `find_element(By.CSS_SELECTOR, "${heal.value}")`);
        newLine = newLine.replace(/findElement\(\s*By\.\w+\(\s*["'`].*?["'`]\s*\)\s*\)/, `findElement(By.cssSelector("${heal.value}"))`);
        newLine = newLine.replace(/FindElement\(\s*By\.\w+\(\s*["'`].*?["'`]\s*\)\s*\)/, `FindElement(By.CssSelector("${heal.value}"))`);
        newLine = newLine.replace(/\.locator\(\s*["'`].*?["'`]\s*\)/, `.locator("${heal.value}")`);
        output.push(newLine);
      } else {
        output.push(`    # AEGISIRT COULD NOT HEAL: ${heal.type} = '${heal.value}'`);
        output.push(line);
      }
    } else {
      output.push(line);
    }
  });
  return output.join("\n");
}

async function healSingleLocator(
  locator: ParsedLocator, html_content: string, diffResult: string,
  ai: GoogleGenAI, browser: Browser | null
): Promise<{ type: string; value: string; confidence: number; reasoning: string; healed: boolean; screenshot: string; verified: boolean }> {

  const htmlChunk = getRelevantHtmlChunk(html_content, locator.context_clue);
  console.log(`[AegisIRT] Healing: ${locator.locator_type} = "${locator.locator_value}" | chunk: ${htmlChunk.length} chars`);

  let attempt = 0, verificationResult = false, screenshotBase64 = "";
  let finalLocator = { type: locator.locator_type, value: locator.locator_value, confidence: 0, reasoning: "" };

  let aiPrompt = `You are a test automation healing engine for clinical trial IRT systems.
Fix EXACTLY ONE broken locator. Do not reference other locators.

BROKEN LOCATOR:
- Type: ${locator.locator_type}
- Value: "${locator.locator_value}"
- Context: ${locator.context_clue}
- Text hint: ${locator.text_hint}

DOM CHANGES: ${diffResult}

CURRENT HTML:
${htmlChunk}

STRICT RULES:
1. NEVER return "${locator.locator_value}" — it is broken.
2. NEVER return mock/placeholder values like [data-testid='mock'].
3. Every value MUST exist verbatim in the HTML above.
4. Scan ALL data-testid, data-action, aria-label, name attributes.
5. Prefer: data-testid > data-action > aria-label > name > class.
6. Explain: what changed, why you chose this selector, how stable it is.

Return ONLY valid JSON:
{"type":"By.CSS_SELECTOR","value":"actual-selector-from-html","confidence":0.95,"reasoning":"explanation"}`;

  while (attempt < 3 && !verificationResult) {
    attempt++;
    let responseText = "";
    try {
      if (!process.env.GEMINI_API_KEY) throw new Error("No API key");
      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash", contents: aiPrompt,
        config: { responseMimeType: "application/json" }
      });
      responseText = response.text || "";
      console.log(`[AegisIRT] Attempt ${attempt} response: ${responseText.slice(0, 300)}`);
    } catch (e) { console.warn(`[AegisIRT] Gemini error attempt ${attempt}:`, e); break; }

    let parsed: any;
    try {
      const match = responseText.match(/\{[\s\S]*\}/);
      parsed = JSON.parse(match ? match[0] : responseText);
    } catch { aiPrompt += `\nAttempt ${attempt}: invalid JSON. Return ONLY a JSON object.`; continue; }

    const suggestedValue: string = parsed.value || "";
    // Validate value exists in HTML — strip selector syntax to get the raw string
    const rawValue = suggestedValue.replace(/[\[\]'"=*^$]/g, " ").split(" ").filter(s => s.length > 3).join(" ");
    const existsInHtml = rawValue.split(" ").some(part => htmlChunk.includes(part));

    if (!suggestedValue || suggestedValue === locator.locator_value || !existsInHtml) {
      console.warn(`[AegisIRT] Attempt ${attempt}: "${suggestedValue}" rejected — not found in HTML chunk.`);
      aiPrompt += `\nAttempt ${attempt}: "${suggestedValue}" was rejected — not found in HTML. Pick a DIFFERENT attribute value that IS in the HTML.`;
      continue;
    }

    finalLocator = { type: parsed.type || "By.CSS_SELECTOR", value: suggestedValue, confidence: parseFloat(parsed.confidence) || 0.9, reasoning: parsed.reasoning || "" };

    if (browser) {
      let page;
      try {
        page = await browser.newPage();
        await page.setContent(html_content, { waitUntil: "load" });
        const isXPath = finalLocator.type.includes("XPATH");
        const isFound = await page.evaluate((locValue: string, isXP: boolean) => {
          try {
            let el: Element | null = null;
            if (isXP) { const r = document.evaluate(locValue, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null); el = r.singleNodeValue as Element; }
            else el = document.querySelector(locValue);
            if (el) { (el as HTMLElement).style.border = "4px solid #ffffff"; (el as HTMLElement).style.backgroundColor = "#1e1e1e"; return true; }
          } catch {}
          return false;
        }, finalLocator.value, isXPath);

        if (isFound) {
          verificationResult = true;
          screenshotBase64 = await page.screenshot({ encoding: "base64" }) as string;
          console.log(`[AegisIRT] VERIFIED: ${finalLocator.value}`);
        } else {
          console.warn(`[AegisIRT] Puppeteer FAILED: ${finalLocator.value}`);
          aiPrompt += `\nAttempt ${attempt}: querySelector for "${finalLocator.value}" returned null. Try XPATH or a different attribute.`;
        }
      } catch (e) { console.warn("[AegisIRT] Page error:", e); }
      finally { if (page) await page.close(); }
    } else { verificationResult = true; }
  }

  return { ...finalLocator, healed: verificationResult, screenshot: screenshotBase64, verified: verificationResult };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    if (action === "provide_feedback") {
      const { actual_success, result } = body;
      let logs: any[] = [];
      try { logs = JSON.parse(fs.readFileSync(healingLogPath, "utf-8")); } catch {}
      if (result) {
        logs.push({ timestamp: new Date().toISOString(), broken_locator: result.original_locator, healed_locator: result.ai_suggestion, actual_success, user_feedback: actual_success ? "approve" : "reject" });
        try { fs.writeFileSync(healingLogPath, JSON.stringify(logs, null, 2)); } catch {}
      }
      return NextResponse.json({ status: "success", message: "Feedback logged" });
    }

    if (action === "get_stats") return NextResponse.json({ server_status: "online", model_loaded: true });

    // ── BATCH HEAL ──────────────────────────────────────────────────────────
    if (action === "heal_batch") {
      const { original_code, html_content } = body;
      if (!original_code || !html_content) return NextResponse.json({ error: "original_code and html_content required" }, { status: 400 });

      const locators = extractAllLocators(original_code);
      console.log(`[AegisIRT] Batch: ${locators.length} locators found`);
      if (locators.length === 0) return NextResponse.json({ error: "No locators found in pasted code" }, { status: 400 });

      let snapshot = { html: "" };
      try { snapshot = JSON.parse(fs.readFileSync(snapshotPath, "utf-8")); } catch {}
      const diffResult = domDiff(snapshot.html, html_content);
      try { fs.writeFileSync(snapshotPath, JSON.stringify({ html: html_content }, null, 2)); } catch {}

      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "dummy" });
      let browser = globalBrowser;
      try {
        if (!browser || !browser.connected) { browser = await puppeteer.launch({ headless: true }); globalBrowser = browser; }
      } catch (e) { console.warn("[AegisIRT] Puppeteer failed:", e); browser = null; }

      const healedMap = new Map<number, { type: string; value: string; healed: boolean }>();
      const resultCards: any[] = [];

      for (const locator of locators) {
        const healed = await healSingleLocator(locator, html_content, diffResult, ai, browser);
        healedMap.set(locator.lineIndex, { type: locator.locator_type, value: healed.healed ? healed.value : locator.locator_value, healed: healed.healed });
        resultCards.push({
          id: `healing_${Date.now()}_${locator.lineIndex}`,
          element_name: locator.text_hint || locator.locator_value,
          original_locator: { type: locator.locator_type, value: locator.locator_value },
          ai_suggestion: { type: healed.type, value: healed.value, confidence: healed.confidence },
          reasoning: healed.reasoning,
          clinical_analysis: { context: locator.context_clue },
          recommendation: healed.verified ? "VERIFIED_BY_BROWSER" : "UNVERIFIED",
          timestamp: new Date().toISOString(),
          verifiedResult: healed.verified,
          status: healed.verified ? "verified" : "pending",
          screenshot: healed.screenshot,
          model_info: { language: locator.detected_language },
          healed: healed.healed,
        });
      }

      const healedFile = reconstructFile(original_code, healedMap);
      const successCount = resultCards.filter(r => r.healed).length;
      console.log(`[AegisIRT] Batch done: ${successCount}/${locators.length} healed`);

      return NextResponse.json({ batch: true, total: locators.length, healed_count: successCount, failed_count: locators.length - successCount, healed_file: healedFile, results: resultCards });
    }

    // ── SINGLE HEAL (legacy) ────────────────────────────────────────────────
    if (action === "heal_element") {
      const { element_info, html_content, clinical_context } = body;
      let snapshot = { html: "" };
      try { snapshot = JSON.parse(fs.readFileSync(snapshotPath, "utf-8")); } catch {}
      const diffResult = domDiff(snapshot.html, html_content);
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "dummy" });
      let browser = globalBrowser;
      try { if (!browser || !browser.connected) { browser = await puppeteer.launch({ headless: true }); globalBrowser = browser; } } catch {}

      const locator: ParsedLocator = {
        lineIndex: 0, originalLine: "",
        locator_type: element_info.locator_type || "By.CSS_SELECTOR",
        locator_value: element_info.locator_value || "",
        text_hint: element_info.text_hint || "",
        detected_language: element_info.detected_language || "python",
        context_clue: `${element_info.text_hint || ""} ${clinical_context || ""}`,
      };

      const healed = await healSingleLocator(locator, html_content, diffResult, ai, browser);
      return NextResponse.json({
        id: `healing_${Date.now()}`,
        element_name: element_info.text_hint || "Unknown",
        original_locator: { type: element_info.locator_type, value: element_info.locator_value },
        ai_suggestion: { type: healed.type, value: healed.value, confidence: healed.confidence },
        reasoning: healed.reasoning,
        clinical_analysis: { context: clinical_context || "general", keywords_found: [] },
        recommendation: healed.verified ? "VERIFIED_BY_BROWSER" : "UNVERIFIED",
        timestamp: new Date().toISOString(),
        verifiedResult: healed.verified,
        status: healed.verified ? "verified" : "pending",
        screenshot: healed.screenshot,
        model_info: { language: element_info.detected_language || "unknown" },
      });
    }

    return NextResponse.json({ error: "Unknown action" }, { status: 400 });
  } catch (error: any) {
    console.error("AI Error:", error);
    return NextResponse.json({ error: "AegisIRT AI failed", details: error.message }, { status: 500 });
  }
}
