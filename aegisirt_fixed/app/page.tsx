"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import { ThumbsUp, ThumbsDown, AlertCircle, CheckCircle, XCircle, Loader2, Shield, Code, BadgeCheck, Copy, Check, ChevronDown, ChevronUp, FileCode } from "lucide-react"
import LandingSection from "@/components/LandingSection"

interface HealingResult {
  id: string
  element_name: string
  original_locator: { type: string; value: string }
  ai_suggestion: { type: string; value: string; confidence: number }
  reasoning: string
  recommendation: string
  timestamp: string
  status: "pending" | "approved" | "rejected" | "verified"
  screenshot?: string
  model_info: { language: string }
  verifiedResult?: boolean
  healed?: boolean
}

export default function AegisIRTSelfHealingAI() {
  const [healingResults, setHealingResults]   = useState<HealingResult[]>([])
  const [isHealing, setIsHealing]             = useState(false)
  const [showLanding, setShowLanding]         = useState(true)
  const [isHowToUseOpen, setIsHowToUseOpen]   = useState(false)
  const [pastedCode, setPastedCode]           = useState("")
  const [htmlContent, setHtmlContent]         = useState("")
  const [healedFile, setHealedFile]           = useState("")
  const [batchSummary, setBatchSummary]       = useState<{ total: number; healed: number; failed: number } | null>(null)
  const [copiedId, setCopiedId]               = useState<string | null>(null)
  const [copiedFile, setCopiedFile]           = useState(false)

  // Detection state
  const [detectionStatus, setDetectionStatus] = useState<{
    language: string; locatorCount: number; context: string;
  } | null>(null)

  // ── Auto-detection parser ──────────────────────────────────────────────────
  useEffect(() => {
    if (!pastedCode) { setDetectionStatus(null); return }

    const code = pastedCode
    const lower = code.toLowerCase()

    // Language
    let language = "undetected"
    if (code.includes("driver.find_element")) language = "Python"
    else if (code.includes("driver.findElement")) language = "Java"
    else if (code.includes("driver.FindElement")) language = "C#"
    else if (code.includes("page.locator") || code.includes("await page.")) language = "TypeScript"

    // Count locators
    const pythonMatches = (code.match(/find_element\(\s*By\.\w+/g) || []).length
    const javaMatches   = (code.match(/findElement\(\s*By\.\w+/g) || []).length
    const csMatches     = (code.match(/FindElement\(\s*By\.\w+/g) || []).length
    const pwMatches     = (code.match(/\.locator\(/g) || []).length
    // Exclude WebDriverWait lines
    const waitMatches   = (code.match(/visibility_of_element_located|WebDriverWait/g) || []).length
    const locatorCount  = Math.max(0, pythonMatches + javaMatches + csMatches + pwMatches - waitMatches)

    // Clinical context
    let context = "undetected"
    if (lower.includes("enroll")) context = "Patient Enrollment"
    else if (lower.includes("randomiz") || lower.includes("stratif")) context = "Randomization"
    else if (lower.includes("dispens") || lower.includes("kit")) context = "Drug Dispensing"
    else if (lower.includes("adverse")) context = "Adverse Events"
    else if (lower.includes("visit")) context = "Visit Management"
    else if (lower.includes("audit") || lower.includes("report")) context = "Audit / Reports"
    else if (lower.includes("amendment")) context = "Protocol Amendments"
    else if (lower.includes("unblind") || lower.includes("emergency")) context = "Emergency Procedures"
    else if (lower.includes("baseline") || lower.includes("updrs")) context = "Baseline Assessments"
    else if (lower.includes("screening") || lower.includes("eligib")) context = "Patient Screening"
    else if (lower.includes("login") || lower.includes("mfa")) context = "Authentication"

    setDetectionStatus({ language, locatorCount, context })
  }, [pastedCode])

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const copyFileToClipboard = () => {
    navigator.clipboard.writeText(healedFile)
    setCopiedFile(true)
    setTimeout(() => setCopiedFile(false), 2000)
  }

  // ── Batch heal ─────────────────────────────────────────────────────────────
  const healElement = async () => {
    if (!pastedCode || !htmlContent) {
      alert("Please paste both broken test code and the current page HTML")
      return
    }

    setIsHealing(true)
    setHealedFile("")
    setBatchSummary(null)
    setHealingResults([])

    try {
      const response = await fetch("/api/ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "heal_batch",
          original_code: pastedCode,
          html_content: htmlContent,
        }),
      })

      if (!response.ok) throw new Error("Healing failed")

      const data = await response.json()

      if (data.batch) {
        setHealingResults(data.results || [])
        setHealedFile(data.healed_file || "")
        setBatchSummary({ total: data.total, healed: data.healed_count, failed: data.failed_count })
      } else {
        // Single result (legacy fallback)
        setHealingResults([data])
      }
    } catch (error) {
      alert(`Healing Failed: ${error instanceof Error ? error.message : "Unknown error"}`)
    } finally {
      setIsHealing(false)
    }
  }

  const provideFeedback = async (resultId: string, feedbackType: "approve" | "reject") => {
    try {
      const result = healingResults.find(r => r.id === resultId)
      if (!result) return
      await fetch("/api/ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "provide_feedback", actual_success: feedbackType === "approve", result }),
      })
      setHealingResults(prev => prev.map(r => r.id === resultId ? { ...r, status: feedbackType === "approve" ? "approved" : "rejected" } : r))
    } catch { alert("Failed to provide feedback") }
  }

  if (showLanding) return <LandingSection onEnter={() => setShowLanding(false)} />

  return (
    <div className="min-h-screen bg-[#080808] text-white p-4" style={{ fontFamily: '"Open Sans", sans-serif' }}>
      <div className="max-w-7xl mx-auto space-y-6">

        {/* Header */}
        <div className="bg-[#0e0e0e] rounded-lg border border-[#1e1e1e] p-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="border border-white p-3 rounded-lg">
              <Shield className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-widest uppercase">AegisIRT</h1>
              <p className="text-[#555] tracking-wide">AI Self-Healing for Clinical Trial Systems</p>
            </div>
          </div>
        </div>

        {/* How to use */}
        <div className="bg-[#0e0e0e] rounded-lg border border-[#1e1e1e] overflow-hidden transition-all duration-300">
          <button
            className="w-full flex items-center justify-between p-3 hover:bg-[#111] transition-colors"
            onClick={() => setIsHowToUseOpen(!isHowToUseOpen)}
          >
            <span className="text-white text-sm font-semibold tracking-wide">How to use AegisIRT</span>
            {isHowToUseOpen ? <ChevronUp className="w-4 h-4 text-white" /> : <ChevronDown className="w-4 h-4 text-white" />}
          </button>
          <div className={`transition-all duration-300 ease-in-out ${isHowToUseOpen ? "max-h-[500px] opacity-100 p-6 pt-4 border-t border-[#1e1e1e]" : "max-h-0 opacity-0 overflow-hidden"}`}>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {[
                { n: 1, title: "Paste Broken Code", desc: "Paste your entire broken Selenium or Playwright test file. Language, locator count, and clinical context are auto-detected instantly." },
                { n: 2, title: "Paste Page DOM", desc: "Paste the current HTML of the page. AegisIRT chunks it intelligently by module so Gemini only sees what it needs." },
                { n: 3, title: "Initialize AI Healing", desc: "Every locator is healed individually by Gemini, then verified in a real headless browser via Puppeteer." },
                { n: 4, title: "Copy Healed File", desc: "Get back the complete test file with all locators replaced. Approve or reject to train the model for next time." },
              ].map(({ n, title, desc }) => (
                <div key={n} className="space-y-3 border-l-2 border-[#1e1e1e] pl-4">
                  <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-white text-black text-xs font-bold">{n}</span>
                  <h3 className="text-white font-semibold text-sm tracking-wide">{title}</h3>
                  <p className="text-[#555] text-sm leading-relaxed">{desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Inputs */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Broken code */}
          <Card className="bg-[#0e0e0e] border-[#1e1e1e] text-white">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2"><Code className="w-5 h-5" /> Paste Broken Test Code</CardTitle>
              <CardDescription className="text-[#555]">Paste your entire test file — all locators are extracted and healed automatically.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                value={pastedCode}
                onChange={(e) => setPastedCode(e.target.value)}
                placeholder={"def test_enroll_patient(driver):\n    driver.find_element(By.ID, 'btn_enroll').click()"}
                className="min-h-[200px] font-mono text-sm bg-[#111] border-[#1e1e1e] text-white focus:border-white focus:ring-1 focus:ring-white transition-all"
              />
              {/* Detection status */}
              {detectionStatus && (
                <p className="text-[#555] text-xs" style={{ fontFamily: '"Open Sans", sans-serif' }}>
                  Detected: <span className="text-white">{detectionStatus.language}</span>
                  {" · "}
                  <span className="text-white">{detectionStatus.locatorCount}</span> locator{detectionStatus.locatorCount !== 1 ? "s" : ""} found
                  {" · "}
                  Context: <span className="text-white">{detectionStatus.context}</span>
                </p>
              )}
            </CardContent>
          </Card>

          {/* HTML */}
          <Card className="bg-[#0e0e0e] border-[#1e1e1e] text-white">
            <CardHeader>
              <CardTitle className="text-white">Current Page DOM</CardTitle>
              <CardDescription className="text-[#555]">Paste the full HTML. AegisIRT chunks it by data-module so large pages work correctly.</CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                value={htmlContent}
                onChange={(e) => setHtmlContent(e.target.value)}
                placeholder={"<div data-module='patient-search'>...</div>"}
                className="min-h-[230px] font-mono text-sm bg-[#111] border-[#1e1e1e] text-white focus:border-white focus:ring-1 focus:ring-white transition-all"
              />
            </CardContent>
          </Card>
        </div>

        {/* Heal button */}
        <div className="flex justify-center">
          <Button
            onClick={healElement}
            disabled={isHealing || !pastedCode || !htmlContent}
            size="lg"
            className="bg-white text-black hover:bg-gray-200 border-2 border-white font-bold px-8 py-6 rounded-none tracking-widest uppercase disabled:opacity-50 disabled:bg-[#111] disabled:text-[#555] disabled:border-[#1e1e1e]"
          >
            {isHealing ? (
              <><Loader2 className="h-5 w-5 mr-3 animate-spin" />Healing {detectionStatus?.locatorCount ?? ""} locators...</>
            ) : (
              <><Shield className="h-5 w-5 mr-3" />Initialize AI Healing Phase</>
            )}
          </Button>
        </div>

        {/* Batch summary */}
        {batchSummary && (
          <div className="bg-[#0e0e0e] border border-[#1e1e1e] rounded-lg p-4 flex items-center gap-6">
            <div className="text-center">
              <p className="text-[#555] text-xs uppercase tracking-wider">Total</p>
              <p className="text-white text-2xl font-bold">{batchSummary.total}</p>
            </div>
            <div className="text-center">
              <p className="text-[#555] text-xs uppercase tracking-wider">Healed</p>
              <p className="text-white text-2xl font-bold">{batchSummary.healed}</p>
            </div>
            <div className="text-center">
              <p className="text-[#555] text-xs uppercase tracking-wider">Failed</p>
              <p className="text-[#555] text-2xl font-bold">{batchSummary.failed}</p>
            </div>
            <div className="flex-1 ml-4">
              <Progress value={(batchSummary.healed / batchSummary.total) * 100} className="h-1 bg-[#111]">
                <div className="h-full bg-white transition-all" style={{ width: `${(batchSummary.healed / batchSummary.total) * 100}%` }} />
              </Progress>
              <p className="text-[#555] text-xs mt-1">{Math.round((batchSummary.healed / batchSummary.total) * 100)}% success rate</p>
            </div>
          </div>
        )}

        {/* Healed file output */}
        {healedFile && (
          <Card className="bg-[#0e0e0e] border-[#1e1e1e] text-white">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-white flex items-center gap-2 uppercase tracking-wider">
                  <FileCode className="w-5 h-5" /> Complete Healed File
                </CardTitle>
                <Button
                  onClick={copyFileToClipboard}
                  className="bg-transparent text-white border border-white hover:bg-white hover:text-black transition-colors uppercase tracking-widest text-xs rounded-none h-8 px-4"
                >
                  {copiedFile ? <><Check className="h-4 w-4 mr-2" />Copied</> : <><Copy className="h-4 w-4 mr-2" />Copy Entire File</>}
                </Button>
              </div>
              <CardDescription className="text-[#555]">Drop this directly back into your test suite.</CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-[#111] border border-[#1e1e1e] p-4 rounded text-xs font-mono text-white overflow-x-auto max-h-[400px] overflow-y-auto whitespace-pre-wrap">
                {healedFile}
              </pre>
            </CardContent>
          </Card>
        )}

        {/* Per-locator result cards */}
        {healingResults.length > 0 && (
          <div className="space-y-4">
            <h2 className="text-xl font-bold uppercase tracking-wider border-b border-[#1e1e1e] pb-2">
              Verified Healing Vectors ({healingResults.filter(r => r.verifiedResult).length}/{healingResults.length})
            </h2>
            {healingResults.map((result) => (
              <Card key={result.id} className={`bg-[#0e0e0e] border text-white ${result.healed === false ? "border-[#333]" : "border-[#1e1e1e]"}`}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 uppercase tracking-wider text-sm">
                      {result.status === "approved" ? <CheckCircle className="h-4 w-4 text-white" />
                        : result.status === "rejected" ? <XCircle className="h-4 w-4 text-[#555]" />
                        : result.verifiedResult ? <CheckCircle className="h-4 w-4 text-white" />
                        : <AlertCircle className="h-4 w-4 text-[#555]" />}
                      {result.element_name}
                    </CardTitle>
                    {result.verifiedResult && (
                      <div className="flex items-center gap-2 bg-[#111] border border-white px-3 py-1 text-xs uppercase tracking-widest font-bold">
                        <BadgeCheck className="w-4 h-4 text-white" /> Browser Verified
                      </div>
                    )}
                    {result.healed === false && (
                      <div className="flex items-center gap-2 bg-[#111] border border-[#333] px-3 py-1 text-xs uppercase tracking-widest font-bold text-[#555]">
                        Could Not Heal
                      </div>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-[#111] border border-[#1e1e1e]">
                      <h4 className="font-bold text-[#555] uppercase text-xs tracking-wider mb-2">Original Broken</h4>
                      <p className="text-sm text-white font-mono break-all">{result.original_locator.type} = "{result.original_locator.value}"</p>
                    </div>
                    <div className={`p-4 bg-[#111] border relative ${result.verifiedResult ? "border-white shadow-[0_0_15px_rgba(255,255,255,0.07)]" : "border-[#333]"}`}>
                      <h4 className="font-bold text-white uppercase text-xs tracking-wider mb-2">Gemini Discovered & Verified Vector</h4>
                      <div className="flex justify-between items-start gap-2">
                        <p className="text-sm text-white font-mono break-all font-bold">{result.ai_suggestion.type} = "{result.ai_suggestion.value}"</p>
                        <Button variant="ghost" size="icon" className="h-6 w-6 text-[#555] hover:text-white hover:bg-[#1e1e1e]"
                          onClick={() => copyToClipboard(result.ai_suggestion.value, result.id)}>
                          {copiedId === result.id ? <Check className="h-4 w-4 text-white" /> : <Copy className="h-4 w-4" />}
                        </Button>
                      </div>
                      {result.reasoning && <p className="text-xs text-[#555] mt-2 italic leading-relaxed">Reasoning: "{result.reasoning}"</p>}
                    </div>
                  </div>

                  {result.screenshot && (
                    <div className="mt-2 border border-[#1e1e1e] p-2 bg-[#111]">
                      <h4 className="font-bold text-[#555] uppercase text-xs tracking-wider mb-2">Browser Verification Render</h4>
                      <img src={`data:image/png;base64,${result.screenshot}`} alt="Highlighted Element" className="w-full max-w-md border border-[#1e1e1e] filter grayscale contrast-125" />
                    </div>
                  )}

                  <div className="flex border-t border-[#1e1e1e] pt-4 mt-4">
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

                  {(result.status === "pending" || result.status === "verified") && (
                    <div className="pt-2 flex gap-3">
                      <Button onClick={() => provideFeedback(result.id, "approve")}
                        className="bg-transparent text-white border border-white hover:bg-white hover:text-black transition-colors uppercase tracking-widest text-xs rounded-none h-10">
                        <ThumbsUp className="h-4 w-4 mr-2" /> Approve
                      </Button>
                      <Button onClick={() => provideFeedback(result.id, "reject")}
                        className="bg-transparent text-[#555] border border-[#1e1e1e] hover:bg-[#1e1e1e] hover:text-white transition-colors uppercase tracking-widest text-xs rounded-none h-10">
                        <ThumbsDown className="h-4 w-4 mr-2" /> Reject
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
