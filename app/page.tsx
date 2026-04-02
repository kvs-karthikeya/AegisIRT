"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ThumbsUp, ThumbsDown, AlertCircle, CheckCircle, XCircle, Loader2, Shield, Code, BadgeCheck, Copy, Check, ChevronDown, ChevronUp } from "lucide-react"
import LandingSection from "@/components/LandingSection"

// Types
interface HealingResult {
  id: string
  element_name: string
  original_locator: { type: string; value: string }
  ai_suggestion: {
    type: string
    value: string
    confidence: number
  }
  reasoning: string
  recommendation: string
  timestamp: string
  status: "pending" | "approved" | "rejected" | "corrected" | "verified"
  screenshot?: string
  model_info: {
    language: string
  }
  verifiedResult?: boolean
}

export default function AegisIRTSelfHealingAI() {
  const [healingResults, setHealingResults] = useState<HealingResult[]>([])
  const [isHealing, setIsHealing] = useState(false)
  const [showLanding, setShowLanding] = useState(true)
  const [isHowToUseOpen, setIsHowToUseOpen] = useState(false)
  
  // Phase 1: Smart input code paste and HTML content
  const [pastedCode, setPastedCode] = useState("")
  const [htmlContent, setHtmlContent] = useState("")
  
  // Auto-populated internal state
  const [parsedInfo, setParsedInfo] = useState({
    locator_type: "",
    locator_value: "",
    tag_name: "",
    text_hint: "",
    detected_language: ""
  })
  
  const [copiedId, setCopiedId] = useState<string | null>(null)

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }
  
  // Parser function that reads pasted Selenium or Playwright test code
  useEffect(() => {
    if (!pastedCode) {
      setParsedInfo({ locator_type: "", locator_value: "", tag_name: "", text_hint: "", detected_language: "" })
      return
    }

    let locator_type = "Unknown"
    let locator_value = ""
    let tag_name = "button"
    let text_hint = ""
    let detected_language = "typescript"

    const code = pastedCode

    // Extract Locator Type
    if (code.includes("By.id") || code.includes("#")) locator_type = "By.ID"
    else if (code.includes("By.xpath") || code.includes("//")) locator_type = "By.XPATH"
    else if (code.includes("By.className") || code.includes("By.cssSelector")) locator_type = "By.CSS_SELECTOR"
    else if (code.includes("getByTestId") || code.includes("data-testid")) locator_type = "By.CSS_SELECTOR"

    // Extract Locator Value
    const stringMatch = code.match(/["'](.*?)["']/)
    if (stringMatch) {
      locator_value = stringMatch[1]
    } else {
      locator_value = code.trim()
    }

    // Extract Language
    if (code.includes("public void") || code.includes("WebElement")) {
      detected_language = "java"
    } else if (code.includes("def ") || code.includes("driver.find_element")) {
      detected_language = "python"
    } else if (code.includes("IWebElement") || code.includes("var ") || code.includes("using OpenQA")) {
      detected_language = "csharp"
    } else {
      detected_language = "typescript"
    }

    // Extract Text Hint based on Context Words
    const lowerCode = code.toLowerCase()
    if (lowerCode.includes("enroll")) text_hint = "Enroll Patient"
    else if (lowerCode.includes("randomize")) text_hint = "Randomize Subject"
    else if (lowerCode.includes("dispense")) text_hint = "Dispense Medication"
    else text_hint = locator_value || "Unknown Text"

    setParsedInfo({ locator_type, locator_value, tag_name, text_hint, detected_language })
  }, [pastedCode])

  const healElement = async () => {
    if (!parsedInfo.locator_value || !htmlContent) {
      alert("Please provide the broken test code and HTML content")
      return
    }

    setIsHealing(true)
    try {
      // Set clinical context dynamically based on text hint
      let clinical_context = "general_irt"
      if (parsedInfo.text_hint === "Enroll Patient") clinical_context = "patient_enrollment"
      else if (parsedInfo.text_hint === "Randomize Subject") clinical_context = "randomization"
      else if (parsedInfo.text_hint === "Dispense Medication") clinical_context = "drug_dispensing"

      const response = await fetch("/api/ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "heal_element",
          element_info: parsedInfo,
          html_content: htmlContent,
          clinical_context: clinical_context
        }),
      })

      if (!response.ok) {
        throw new Error("Healing failed")
      }

      const result = await response.json()
      setHealingResults((prev) => [result, ...prev])
    } catch (error) {
      alert(`Healing Failed: ${error instanceof Error ? error.message : "Unknown error"}`)
    } finally {
      setIsHealing(false)
    }
  }

  const provideFeedback = async (resultId: string, feedbackType: "approve" | "reject") => {
    try {
      const healingIndex = healingResults.findIndex((r) => r.id === resultId)
      if (healingIndex === -1) return

      const result = healingResults[healingIndex]

      await fetch("/api/ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "provide_feedback",
          actual_success: feedbackType === "approve",
          result: result
        }),
      })

      setHealingResults((prev) =>
        prev.map((r) =>
          r.id === resultId ? { ...r, status: feedbackType === "approve" ? "approved" : "rejected" } : r
        )
      )
    } catch (error) {
      alert("Failed to provide feedback")
    }
  }

  // Pure dark styling applied explicitly
  if (showLanding) {
    return <LandingSection onEnter={() => setShowLanding(false)} />
  }

  return (
    <div className="min-h-screen bg-[#080808] text-white p-4 font-sans" style={{ fontFamily: '"Open Sans", sans-serif' }}>
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-[#0e0e0e] rounded-lg border border-[#1e1e1e] p-6 flex flex-col md:flex-row items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="border border-white p-3 rounded-lg flex items-center justify-center">
              <Shield className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-widest uppercase">AegisIRT</h1>
              <p className="text-[#555] tracking-wide">AI Self-Healing for Clinical Trial Systems</p>
            </div>
          </div>
        </div>

        {/* How to use panel */}
        <div className="bg-[#0e0e0e] rounded-lg border border-[#1e1e1e] overflow-hidden font-sans transition-all duration-300">
          <button 
            className="w-full flex items-center justify-between p-3 bg-[#0e0e0e] hover:bg-[#111] transition-colors"
            onClick={() => setIsHowToUseOpen(!isHowToUseOpen)}
          >
            <span className="text-white text-sm font-semibold tracking-wide flex items-center gap-2">How to use AegisIRT</span>
            {isHowToUseOpen ? <ChevronUp className="w-4 h-4 text-white" /> : <ChevronDown className="w-4 h-4 text-white" />}
          </button>
          
          <div className={`transition-all duration-300 ease-in-out ${isHowToUseOpen ? 'max-h-[500px] opacity-100 p-6 pt-4 border-t border-[#1e1e1e]' : 'max-h-0 opacity-0 overflow-hidden'}`}>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="space-y-3 border-l-2 border-[#1e1e1e] pl-4">
                <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-white text-black text-xs font-bold">1</span>
                <h3 className="text-white font-semibold text-sm tracking-wide">Paste Code</h3>
                <p className="text-[#555] text-sm leading-relaxed">Paste your broken Selenium or Playwright test code into the input panel.</p>
              </div>
              <div className="space-y-3 border-l-2 border-[#1e1e1e] pl-4">
                <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-white text-black text-xs font-bold">2</span>
                <h3 className="text-white font-semibold text-sm tracking-wide">Select Context</h3>
                <p className="text-[#555] text-sm leading-relaxed">Select your clinical context and output language.</p>
              </div>
              <div className="space-y-3 border-l-2 border-[#1e1e1e] pl-4">
                <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-white text-black text-xs font-bold">3</span>
                <h3 className="text-white font-semibold text-sm tracking-wide">Heal Vector</h3>
                <p className="text-[#555] text-sm leading-relaxed">Click Heal — Gemini analyzes your DOM and suggests a fix which Puppeteer then verifies in a real browser.</p>
              </div>
              <div className="space-y-3 border-l-2 border-[#1e1e1e] pl-4">
                <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-white text-black text-xs font-bold">4</span>
                <h3 className="text-white font-semibold text-sm tracking-wide">Verify & Train</h3>
                <p className="text-[#555] text-sm leading-relaxed">Approve or reject the result to train the model for future heals.</p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Code Paste Section */}
          <Card className="bg-[#0e0e0e] border-[#1e1e1e] text-white">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2"><Code className="w-5 h-5"/> Paste Broken Locator Code</CardTitle>
              <CardDescription className="text-[#555]">
                Paste your Selenium or Playwright test code. We auto-extract locator bindings and context.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                value={pastedCode}
                onChange={(e) => setPastedCode(e.target.value)}
                placeholder={'Example: await page.locator("getByTestId(\'btn_enroll\')").click();'}
                className="min-h-[150px] font-mono text-sm bg-[#111] border-[#1e1e1e] text-white focus:border-white focus:ring-1 focus:ring-white transition-all"
              />
              
              {/* Auto-populated badges */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-2">
                <div className="bg-[#111] border border-[#1e1e1e] p-2 flex flex-col rounded">
                  <span className="text-xs text-[#555] uppercase tracking-wider">Type</span>
                  <span className="text-sm font-semibold">{parsedInfo.locator_type || "—"}</span>
                </div>
                <div className="bg-[#111] border border-[#1e1e1e] p-2 flex flex-col rounded truncate">
                  <span className="text-xs text-[#555] uppercase tracking-wider">Value</span>
                  <span className="text-sm font-semibold truncate" title={parsedInfo.locator_value}>{parsedInfo.locator_value || "—"}</span>
                </div>
                <div className="bg-[#111] border border-[#1e1e1e] p-2 flex flex-col rounded">
                  <span className="text-xs text-[#555] uppercase tracking-wider">Language</span>
                  <span className="text-sm font-semibold">{parsedInfo.detected_language || "—"}</span>
                </div>
                <div className="bg-[#111] border border-[#1e1e1e] p-2 flex flex-col rounded truncate">
                   <span className="text-xs text-[#555] uppercase tracking-wider">Hint</span>
                  <span className="text-sm font-semibold truncate" title={parsedInfo.text_hint}>{parsedInfo.text_hint || "—"}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* HTML Content */}
          <Card className="bg-[#0e0e0e] border-[#1e1e1e] text-white">
            <CardHeader>
              <CardTitle className="text-white">Current Page DOM</CardTitle>
              <CardDescription className="text-[#555]">Context DOM for AI analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                value={htmlContent}
                onChange={(e) => setHtmlContent(e.target.value)}
                placeholder="<div id='app'>...</div>"
                className="min-h-[210px] font-mono text-sm bg-[#111] border-[#1e1e1e] text-white focus:border-white focus:ring-1 focus:ring-white transition-all"
              />
            </CardContent>
          </Card>
        </div>

        {/* Heal Button */}
        <div className="flex justify-center mb-6">
          <Button
            onClick={healElement}
            disabled={isHealing || !parsedInfo.locator_value || !htmlContent}
            size="lg"
            className="bg-white text-black hover:bg-gray-200 border-2 border-white transition-all font-bold px-8 py-6 rounded-none tracking-widest uppercase disabled:opacity-50 disabled:bg-[#111] disabled:text-[#555] disabled:border-[#1e1e1e]"
          >
            {isHealing ? (
              <>
                <Loader2 className="h-5 w-5 mr-3 animate-spin" />
                Processing Gemini Multi-modal Analysis...
              </>
            ) : (
              <>
                <Shield className="h-5 w-5 mr-3" />
                Initialize AI Healing Phase
              </>
            )}
          </Button>
        </div>

        {/* Results */}
        {healingResults.length > 0 && (
          <div className="space-y-6">
            <h2 className="text-xl font-bold uppercase tracking-wider border-b border-[#1e1e1e] pb-2">Verified Healing Vectors</h2>
            {healingResults.map((result) => (
              <Card key={result.id} className="bg-[#0e0e0e] border-[#1e1e1e] text-white">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 uppercase tracking-wider">
                      {result.status === "approved" ? (
                        <CheckCircle className="h-5 w-5 text-white" />
                      ) : result.status === "rejected" ? (
                        <XCircle className="h-5 w-5 text-[#555]" />
                      ) : (
                        <AlertCircle className="h-5 w-5 text-white" />
                      )}
                      {result.element_name}
                    </CardTitle>
                    {result.status === "verified" || result.verifiedResult ? (
                       <div className="flex items-center gap-2 bg-[#111] border border-white px-3 py-1 text-xs uppercase tracking-widest font-bold">
                         <BadgeCheck className="w-4 h-4 text-white" /> Browser Verified
                       </div>
                    ) : null}
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-[#111] border border-[#1e1e1e]">
                      <h4 className="font-bold text-[#555] uppercase text-xs tracking-wider mb-2">Original Broken Input</h4>
                      <p className="text-sm text-white font-mono break-all">
                        {result.original_locator.type} = "{result.original_locator.value}"
                      </p>
                    </div>

                    <div className="p-4 bg-[#111] border border-white shadow-[0_0_15px_rgba(255,255,255,0.1)] relative group">
                      <h4 className="font-bold text-white uppercase text-xs tracking-wider mb-2">Gemini Discovered & Verified Vector</h4>
                      <div className="flex justify-between items-start gap-2">
                        <p className="text-sm text-white font-mono break-all font-bold">
                          {result.ai_suggestion.type} = "{result.ai_suggestion.value}"
                        </p>
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          className="h-6 w-6 text-[#555] hover:text-white hover:bg-[#1e1e1e]"
                          onClick={() => copyToClipboard(result.ai_suggestion.value, result.id)}
                          title="Copy Locator Value"
                        >
                          {copiedId === result.id ? <Check className="h-4 w-4 text-white" /> : <Copy className="h-4 w-4" />}
                        </Button>
                      </div>
                      <p className="text-xs text-[#555] mt-2 italic">Reasoning: "{result.reasoning}"</p>
                    </div>
                  </div>

                  {result.screenshot && (
                     <div className="mt-4 border border-[#1e1e1e] p-2 bg-[#111]">
                       <h4 className="font-bold text-[#555] uppercase text-xs tracking-wider mb-2">Browser Verification Render</h4>
                       <img src={`data:image/png;base64,${result.screenshot}`} alt="Highlighted Element" className="w-full max-w-md border border-[#1e1e1e] filter grayscale contrast-125" />
                     </div>
                  )}

                  <div className="flex border-t border-[#1e1e1e] pt-4 mt-6">
                    <div className="flex-1">
                      <h4 className="font-semibold text-xs tracking-wider uppercase mb-2">Model Confidence</h4>
                      <div className="flex items-center gap-3">
                        <Progress value={result.ai_suggestion.confidence * 100} className="h-1 bg-[#111] flex-1">
                          <div className="h-full bg-white transition-all" style={{ width: `${result.ai_suggestion.confidence * 100}%`}} />
                        </Progress>
                        <p className="text-xs font-mono">{(result.ai_suggestion.confidence * 100).toFixed(1)}%</p>
                      </div>
                    </div>
                  </div>

                  {(result.status === "pending" || result.status === "verified") && (
                    <div className="pt-4 flex gap-3">
                      <Button
                        onClick={() => provideFeedback(result.id, "approve")}
                        className="bg-transparent text-white border border-white hover:bg-white hover:text-black transition-colors uppercase tracking-widest text-xs rounded-none h-10"
                      >
                        <ThumbsUp className="h-4 w-4 mr-2" />
                        Approve Mapping
                      </Button>
                      <Button 
                        onClick={() => provideFeedback(result.id, "reject")} 
                        className="bg-transparent text-[#555] border border-[#1e1e1e] hover:bg-[#1e1e1e] hover:text-white transition-colors uppercase tracking-widest text-xs rounded-none h-10"
                      >
                        <ThumbsDown className="h-4 w-4 mr-2" />
                        Reject
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
