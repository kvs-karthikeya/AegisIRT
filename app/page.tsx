"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import { AlertCircle, CheckCircle, XCircle, Loader2, Shield, Code, BadgeCheck, Copy, Check, ChevronDown, ChevronUp, ThumbsUp, ThumbsDown } from "lucide-react"
import LandingSection from "@/components/LandingSection"

interface HealingResult {
  id: string
  element_name: string
  original_locator: { type: string; value: string }
  ai_suggestion: {
    type: string
    value: string
    replacement_code?: string
    confidence: number
  }
  reasoning: string
  status: "pending" | "approved" | "rejected" | "unverified" | "verified"
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

  const [pastedCode, setPastedCode] = useState("")
  const [htmlContent, setHtmlContent] = useState("")
  const [healedCode, setHealedCode] = useState("")

  const [detectedLanguage, setDetectedLanguage] = useState("")
  const [locatorCount, setLocatorCount] = useState(0)
  const [detectedContext, setDetectedContext] = useState("")

  const [copiedId, setCopiedId] = useState<string | null>(null)

  const handleTestCodeChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value
    setPastedCode(val)

    if (!val) {
      setDetectedLanguage("")
      setLocatorCount(0)
      setDetectedContext("")
      return
    }

    let lang = "undetected"
    let count = 0
    if (val.includes("driver.find_element(By.")) {
      lang = "Python"
      count = (val.match(/driver\.find_element\(By\./g) || []).length
    } else if (val.includes("driver.FindElement(By.")) {
      lang = "C#"
      count = (val.match(/driver\.FindElement\(By\./g) || []).length
    } else if (val.includes("driver.findElement(By.")) {
      lang = "Java"
      count = (val.match(/driver\.findElement\(By\./g) || []).length
    } else if (val.includes("page.locator(") || val.includes("await page.")) {
      lang = "TypeScript"
      count = (val.match(/page\.locator\(/g) || []).length + (val.match(/await page\./g) || []).length
    }
    setDetectedLanguage(lang)
    setLocatorCount(count)

    const lowerCode = val.toLowerCase()
    let ctx = "undetected"
    if (lowerCode.includes("enroll") || lowerCode.includes("patient_search")) ctx = "Patient Enrollment"
    else if (lowerCode.includes("randomiz")) ctx = "Randomization"
    else if (lowerCode.includes("dispens") || lowerCode.includes("kit")) ctx = "Drug Dispensing"
    else if (lowerCode.includes("adverse")) ctx = "Adverse Events"
    else if (lowerCode.includes("visit")) ctx = "Visit Management"
    
    setDetectedContext(ctx)
  }

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const healBatch = async () => {
    if (!pastedCode || !htmlContent) {
      alert("Please provide the broken test code and HTML content")
      return
    }

    setIsHealing(true)
    try {
      let clinical_context = "general_irt"
      const lowerCode = pastedCode.toLowerCase()
      if (lowerCode.includes("enroll")) clinical_context = "patient_enrollment"
      else if (lowerCode.includes("randomize")) clinical_context = "randomization"
      else if (lowerCode.includes("dispense")) clinical_context = "drug_dispensing"

      const response = await fetch("/api/ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "heal_batch",
          test_code: pastedCode,
          html_content: htmlContent,
          clinical_context: clinical_context
        }),
      })

      if (!response.ok) {
        throw new Error("Healing failed")
      }

      const data = await response.json()
      setHealingResults(data.results)
      setHealedCode(data.healed_test_code)
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
                <h3 className="text-white font-semibold text-sm tracking-wide">Paste Batch Code</h3>
                <p className="text-[#555] text-sm leading-relaxed">Paste your entire broken test script with multiple locators into the first panel.</p>
              </div>
              <div className="space-y-3 border-l-2 border-[#1e1e1e] pl-4">
                <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-white text-black text-xs font-bold">2</span>
                <h3 className="text-white font-semibold text-sm tracking-wide">Paste Page DOM</h3>
                <p className="text-[#555] text-sm leading-relaxed">Paste the HTML structure (DOM) of the page into the second panel.</p>
              </div>
              <div className="space-y-3 border-l-2 border-[#1e1e1e] pl-4">
                <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-white text-black text-xs font-bold">3</span>
                <h3 className="text-white font-semibold text-sm tracking-wide">Batch AI Healing</h3>
                <p className="text-[#555] text-sm leading-relaxed">Gemini isolates and analyzes each locator, replacing them with syntactically exact verified fixes.</p>
              </div>
              <div className="space-y-3 border-l-2 border-[#1e1e1e] pl-4">
                <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-white text-black text-xs font-bold">4</span>
                <h3 className="text-white font-semibold text-sm tracking-wide">Copy Healed File</h3>
                <p className="text-[#555] text-sm leading-relaxed">Review individual component verifications, then copy the fully composed healed script back.</p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <Card className="bg-[#0e0e0e] border-[#1e1e1e] text-white">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2"><Code className="w-5 h-5" /> Paste Test Code</CardTitle>
              <CardDescription className="text-[#555]">
                Paste your comprehensive Selenium or Playwright test code containing broken locators.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                value={pastedCode}
                onChange={handleTestCodeChange}
                placeholder={'await page.locator("getByTestId(\'btn_enroll\')").click();\nawait page.locator(".old-class").fill("data");'}
                className="min-h-[200px] font-mono text-sm bg-[#111] border-[#1e1e1e] text-white focus:border-white focus:ring-1 focus:ring-white transition-all"
              />
              {pastedCode && (
                <div className="mt-2 text-[#555] text-[12px] font-sans">
                  Detected: {detectedLanguage} &middot; {locatorCount} locators found &middot; Context: {detectedContext}
                </div>
              )}
            </CardContent>
          </Card>

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
                className="min-h-[200px] font-mono text-sm bg-[#111] border-[#1e1e1e] text-white focus:border-white focus:ring-1 focus:ring-white transition-all"
              />
            </CardContent>
          </Card>
        </div>

        <div className="flex justify-center mb-6">
          <Button
            onClick={healBatch}
            disabled={isHealing || !pastedCode || !htmlContent}
            size="lg"
            className="bg-white text-black hover:bg-gray-200 border-2 border-white transition-all font-bold px-8 py-6 rounded-none tracking-widest uppercase disabled:opacity-50 disabled:bg-[#111] disabled:text-[#555] disabled:border-[#1e1e1e]"
          >
            {isHealing ? (
              <>
                <Loader2 className="h-5 w-5 mr-3 animate-spin" />
                Processing Batch Healing Array...
              </>
            ) : (
              <>
                <Shield className="h-5 w-5 mr-3" />
                Execute Global Healing Sequence
              </>
            )}
          </Button>
        </div>

        {healedCode && (
           <Card className="bg-[#0b0b0b] border-white shadow-[0_0_20px_rgba(255,255,255,0.05)] text-white mb-6">
             <CardHeader className="flex flex-row items-center justify-between border-b border-[#1e1e1e] pb-4">
               <div>
                 <CardTitle className="text-white uppercase tracking-wider flex items-center gap-2">
                   Fully Healed Test Script
                 </CardTitle>
                 <CardDescription className="text-[#555]">
                   Complete file output with all broken locators replaced automatically.
                 </CardDescription>
               </div>
               <Button
                 onClick={() => copyToClipboard(healedCode, "full_code")}
                 className="bg-white text-black hover:bg-gray-200 transition-colors uppercase tracking-widest text-xs rounded-none h-10 font-bold"
               >
                 {copiedId === "full_code" ? <Check className="h-4 w-4 mr-2" /> : <Copy className="h-4 w-4 mr-2" />}
                 Copy File
               </Button>
             </CardHeader>
             <CardContent className="pt-6">
                <Textarea
                  readOnly
                  value={healedCode}
                  className="min-h-[250px] font-mono text-sm bg-black border border-[#1e1e1e] text-green-400 focus-visible:ring-0 selection:bg-[#333] cursor-text"
                />
             </CardContent>
           </Card>
        )}

        {healingResults.length > 0 && (
          <div className="space-y-6">
            <h2 className="text-xl font-bold uppercase tracking-wider border-b border-[#1e1e1e] pb-2">Isolated Element Validations</h2>
            {healingResults.map((result) => (
              <Card key={result.id} className="bg-[#0e0e0e] border-[#1e1e1e] text-white">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base flex items-center gap-2 tracking-wider font-mono bg-[#111] py-1 px-3 border border-[#1e1e1e]">
                      {result.status === "approved" ? (
                        <CheckCircle className="h-4 w-4 text-white" />
                      ) : result.status === "rejected" ? (
                        <XCircle className="h-4 w-4 text-[#555]" />
                      ) : (
                        <AlertCircle className="h-4 w-4 text-white" />
                      )}
                      {result.element_name}
                    </CardTitle>
                    {result.status === "verified" || result.verifiedResult ? (
                      <div className="flex items-center gap-2 bg-[#111] border border-white px-3 py-1 text-xs uppercase tracking-widest font-bold">
                        <BadgeCheck className="w-4 h-4 text-white" /> Browser Verified
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 bg-[#111] border border-[#333] text-[#555] px-3 py-1 text-xs uppercase tracking-widest font-bold">
                        <XCircle className="w-4 h-4 text-[#555]" /> Unverified
                      </div>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-[#111] border border-[#1e1e1e]">
                      <h4 className="font-bold text-[#555] uppercase text-xs tracking-wider mb-2">Original Broken Input</h4>
                      <p className="text-sm text-white font-mono break-all font-semibold">
                        {result.original_locator.value}
                      </p>
                    </div>

                    <div className="p-4 bg-[#111] border border-white shadow-[0_0_15px_rgba(255,255,255,0.05)] relative group">
                      <h4 className="font-bold text-white uppercase text-xs tracking-wider mb-2">Gemini Discovered Vector</h4>
                      <div className="flex justify-between items-start gap-2">
                        <p className="text-sm text-white font-mono break-all font-bold">
                          {result.ai_suggestion.replacement_code || result.ai_suggestion.value}
                        </p>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 text-[#555] hover:text-white hover:bg-[#1e1e1e]"
                          onClick={() => copyToClipboard(result.ai_suggestion.replacement_code || result.ai_suggestion.value, result.id)}
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
                          <div className="h-full bg-white transition-all" style={{ width: `${result.ai_suggestion.confidence * 100}%` }} />
                        </Progress>
                        <p className="text-xs font-mono">{(result.ai_suggestion.confidence * 100).toFixed(1)}%</p>
                      </div>
                    </div>
                  </div>

                  {(result.status === "unverified" || result.status === "verified" || result.status === "pending") && (
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