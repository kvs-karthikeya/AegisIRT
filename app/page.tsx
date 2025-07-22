"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ThumbsUp, ThumbsDown, Edit3, Send, AlertCircle, CheckCircle, XCircle, Loader2, Shield } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Linkedin, Mail, Github } from "lucide-react"

interface HealingResult {
  id: string
  element_name: string
  original_locator: { type: string; value: string }
  ai_suggestion: {
    type: string
    value: string
    confidence: number
    language_specific: {
      language: string
      framework: string
      locator_method: string
      code_snippet: string
      imports: string[]
    }
  }
  clinical_analysis: { context: string; keywords_found: string[] }
  recommendation: string
  timestamp: string
  status: "pending" | "approved" | "rejected" | "corrected"
  model_info: {
    language: string
    framework: string
  }
}

interface AIStats {
  server_status: string
  model_loaded: boolean
  total_predictions: number
  final_accuracy?: number
  parameters?: number
  supported_languages?: string[]
  multi_language_support?: boolean
}

export default function AegisIRTSelfHealingAI() {
  const [healingResults, setHealingResults] = useState<HealingResult[]>([])
  const [aiStats, setAIStats] = useState<AIStats | null>(null)
  const [supportedLanguages, setSupportedLanguages] = useState<any[]>([])
  const [isHealing, setIsHealing] = useState(false)
  const [correctionMode, setCorrectionMode] = useState(false)
  const [selectedResult, setSelectedResult] = useState<HealingResult | null>(null)
  const [correctedType, setCorrectedType] = useState("")
  const [correctedValue, setCorrectedValue] = useState("")

  // Form state
  const [elementInfo, setElementInfo] = useState({
    id: "",
    name: "",
    class: "",
    text_hint: "",
    tag_name: "button",
    locator_type: "By.ID",
    locator_value: "",
  })
  const [htmlContent, setHtmlContent] = useState("")
  const [clinicalContext, setClinicalContext] = useState("")
  const [selectedLanguage, setSelectedLanguage] = useState("java")

  useEffect(() => {
    loadAIStats()
    loadSupportedLanguages()
  }, [])

  const loadAIStats = async () => {
    try {
      const response = await fetch("/api/ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "get_stats" }),
      })
      const stats = await response.json()
      setAIStats(stats)
    } catch (error) {
      setAIStats({
        server_status: "offline",
        model_loaded: false,
        total_predictions: 0,
      })
    }
  }

  const loadSupportedLanguages = async () => {
    try {
      const response = await fetch("/api/ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "get_supported_languages" }),
      })
      const data = await response.json()
      setSupportedLanguages(data.languages || [])
    } catch (error) {
      console.error("Failed to load supported languages")
    }
  }

  const healElement = async () => {
    if (!elementInfo.text_hint || !htmlContent) {
      alert("Please provide element text hint and HTML content")
      return
    }

    setIsHealing(true)
    try {
      const response = await fetch("/api/ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "heal_element",
          element_info: elementInfo,
          html_content: htmlContent,
          clinical_context: clinicalContext,
          language: selectedLanguage,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.details || "Healing failed")
      }

      const result = await response.json()

      const healingResult: HealingResult = {
        id: `healing_${Date.now()}`,
        element_name: result.element_name,
        original_locator: result.original_locator,
        ai_suggestion: result.ai_suggestion,
        clinical_analysis: result.clinical_analysis,
        recommendation: result.recommendation,
        timestamp: result.timestamp,
        status: "pending",
        model_info: result.model_info,
      }

      setHealingResults((prev) => [healingResult, ...prev])
      loadAIStats()
    } catch (error) {
      alert(`AegisIRT Healing Failed: ${error instanceof Error ? error.message : "Unknown error"}`)
    } finally {
      setIsHealing(false)
    }
  }

  const provideFeedback = async (resultId: string, feedbackType: "approve" | "reject" | "correct") => {
    try {
      const healingIndex = healingResults.findIndex((r) => r.id === resultId)
      if (healingIndex === -1) return

      const result = healingResults[healingIndex]

      await fetch("/api/ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "provide_feedback",
          healing_id: healingIndex,
          actual_success: feedbackType === "approve" || feedbackType === "correct",
          language: result.model_info.language,
        }),
      })

      setHealingResults((prev) =>
        prev.map((result) =>
          result.id === resultId
            ? {
                ...result,
                status: feedbackType === "correct" ? "corrected" : feedbackType === "approve" ? "approved" : "rejected",
              }
            : result,
        ),
      )

      loadAIStats()
    } catch (error) {
      alert("Failed to provide feedback")
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    alert("Code copied to clipboard!")
  }

  const startCorrection = (result: HealingResult) => {
    setSelectedResult(result)
    setCorrectionMode(true)
    setCorrectedType(result.ai_suggestion.type)
    setCorrectedValue(result.ai_suggestion.value)
  }

  const submitCorrection = () => {
    if (!selectedResult || !correctedValue) return
    provideFeedback(selectedResult.id, "correct")
    setCorrectionMode(false)
    setSelectedResult(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-100 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border mb-6 p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="bg-gradient-to-r from-slate-700 to-blue-600 p-3 rounded-lg">
                <Shield className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">AegisIRT</h1>
                <p className="text-gray-600">Multi-Language AI Self-Healing for Clinical Trial Systems</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              {aiStats && (
                <>
                  <div className="text-right">
                    <p className="text-sm text-gray-600">AegisIRT Status</p>
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-2 h-2 rounded-full ${aiStats.server_status === "online" ? "bg-green-500" : "bg-red-500"}`}
                      />
                      <p className="text-sm font-semibold">
                        {aiStats.server_status === "online" ? "Neural Network Online" : "AI Offline"}
                      </p>
                    </div>
                  </div>
                  {aiStats.final_accuracy && (
                    <div className="text-right">
                      <p className="text-sm text-gray-600">Model Accuracy</p>
                      <p className="text-xl font-bold text-green-600">{(aiStats.final_accuracy * 100).toFixed(1)}%</p>
                    </div>
                  )}
                  {aiStats.multi_language_support && (
                    <div className="text-right">
                      <p className="text-sm text-gray-600">Languages</p>
                      <p className="text-sm font-semibold">{supportedLanguages.length} Supported</p>
                    </div>
                  )}
                </>
              )}
              {/* About Me Button and Dialog */}
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="outline" className="ml-4 bg-transparent">
                    About Me
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-[600px]">
                  <DialogHeader>
                    <DialogTitle>About K.V.S.K</DialogTitle>
                    <DialogDescription>
                      My academic journey is centered around the powerful intersection of biotechnology, computer
                      science, and artificial intelligence, with a strong belief in their potential to transform the
                      life sciences.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="py-4 text-gray-700 text-sm leading-relaxed">
                    <p className="mb-2">
                      {
                        "I’m a highly motivated sophomore pursuing a B.Tech. in Biotechnology (Hons.) with a specialization in Bioinformatics at KL University, complemented by a Minor in CSE and Advanced Technologies from IIT Mandi. My academic journey is centered around the powerful intersection of biotechnology, computer science, and artificial intelligence, with a strong belief in their potential to transform the life sciences."
                      }
                    </p>
                    <p className="mb-2">
                      {
                        "My passion extends beyond the classroom—I’ve led and contributed to impactful initiatives such as “My Critters” and “smart.bucks,” participated in national-level hackathons and ideathons, and served as a lead in the Vachas Club, where I refined my skills in team-building, communication, and project execution."
                      }
                    </p>
                    <p>
                      {
                        "I’m currently focused on developing expertise in SAS, Clinical Data Management, Data Science, Statistical Programming, and AI/ML. With a growth mindset and entrepreneurial spirit, I’m committed to driving innovation that bridges science and technology to solve real-world challenges.Let’s connect and build something meaningful together!"
                      }
                    </p>
                  </div>
                  <div className="flex justify-center gap-6 pt-4 border-t">
                    <a
                      href="https://www.linkedin.com/in/your-linkedin-profile"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800 flex items-center gap-2"
                    >
                      <Linkedin className="h-5 w-5" /> LinkedIn
                    </a>
                    <a
                      href="mailto:your.email@example.com"
                      className="text-red-600 hover:text-red-800 flex items-center gap-2"
                    >
                      <Mail className="h-5 w-5" /> Email
                    </a>
                    <a
                      href="https://github.com/your-github-profile"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-gray-800 hover:text-gray-900 flex items-center gap-2"
                    >
                      <Github className="h-5 w-5" /> GitHub
                    </a>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </div>

        {/* Language Support Alert */}
        {/* AI Status Alert */}
        {aiStats && aiStats.server_status !== "online" && (
          <Alert className="mb-6 border-red-200 bg-red-50">
            <AlertCircle className="h-4 w-4 text-red-600" />
            <AlertDescription className="text-red-800">
              AegisIRT AI model is offline or failed to load. Check console for errors.
            </AlertDescription>
          </Alert>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Element Input */}
          <Card>
            <CardHeader>
              <CardTitle>Broken IRT Element</CardTitle>
              <CardDescription>
                Provide details about the broken clinical trial element for AegisIRT to heal
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="clinical-context">Clinical Context</Label>
                  <Select value={clinicalContext} onValueChange={setClinicalContext}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select context" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="patient_enrollment">Patient Enrollment</SelectItem>
                      <SelectItem value="randomization">Randomization</SelectItem>
                      <SelectItem value="drug_dispensing">Drug Dispensing</SelectItem>
                      <SelectItem value="adverse_events">Adverse Events</SelectItem>
                      <SelectItem value="visit_management">Visit Management</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="element-id">Element ID</Label>
                  <Input
                    id="element-id"
                    value={elementInfo.id}
                    onChange={(e) => setElementInfo({ ...elementInfo, id: e.target.value })}
                    placeholder="btn_enroll_patient"
                  />
                </div>
                <div>
                  <Label htmlFor="element-name">Element Name</Label>
                  <Input
                    id="element-name"
                    value={elementInfo.name}
                    onChange={(e) => setElementInfo({ ...elementInfo, name: e.target.value })}
                    placeholder="enroll_patient"
                  />
                </div>
              </div>

              <div>
                <Label htmlFor="element-class">Element Class</Label>
                <Input
                  id="element-class"
                  value={elementInfo.class}
                  onChange={(e) => setElementInfo({ ...elementInfo, class: e.target.value })}
                  placeholder="btn btn-primary enrollment-btn"
                />
              </div>

              <div>
                <Label htmlFor="text-hint">Text Hint (Required)</Label>
                <Input
                  id="text-hint"
                  value={elementInfo.text_hint}
                  onChange={(e) => setElementInfo({ ...elementInfo, text_hint: e.target.value })}
                  placeholder="Enroll Patient"
                  className="border-blue-300"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="tag-name">Tag Name</Label>
                  <Select
                    value={elementInfo.tag_name}
                    onValueChange={(value) => setElementInfo({ ...elementInfo, tag_name: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="button">button</SelectItem>
                      <SelectItem value="input">input</SelectItem>
                      <SelectItem value="select">select</SelectItem>
                      <SelectItem value="div">div</SelectItem>
                      <SelectItem value="span">span</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label htmlFor="locator-type">Broken Locator Type</Label>
                  <Select
                    value={elementInfo.locator_type}
                    onValueChange={(value) => setElementInfo({ ...elementInfo, locator_type: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="By.ID">By.ID</SelectItem>
                      <SelectItem value="By.NAME">By.NAME</SelectItem>
                      <SelectItem value="By.CLASS_NAME">By.CLASS_NAME</SelectItem>
                      <SelectItem value="By.XPATH">By.XPATH</SelectItem>
                      <SelectItem value="By.CSS_SELECTOR">By.CSS_SELECTOR</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <Label htmlFor="locator-value">Broken Locator Value</Label>
                <Input
                  id="locator-value"
                  value={elementInfo.locator_value}
                  onChange={(e) => setElementInfo({ ...elementInfo, locator_value: e.target.value })}
                  placeholder="btn_enroll_patient"
                />
              </div>
            </CardContent>
          </Card>

          {/* HTML Content */}
          <Card>
            <CardHeader>
              <CardTitle>Current Page Content</CardTitle>
              <CardDescription>Page content for AegisIRT neural network analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                value={htmlContent}
                onChange={(e) => setHtmlContent(e.target.value)}
                placeholder="<div class='patient-enrollment'>&#10;  <button data-testid='enroll-patient-action'>Enroll Patient</button>&#10;</div>"
                className="min-h-[300px] font-mono text-sm border-blue-300"
              />
            </CardContent>
          </Card>
        </div>

        {/* Heal Button */}
        <div className="text-center mb-6">
          <Button
            onClick={healElement}
            disabled={isHealing || !elementInfo.text_hint || !htmlContent}
            size="lg"
            className="bg-gradient-to-r from-slate-700 to-blue-600 hover:from-slate-800 hover:to-blue-700"
          >
            {isHealing ? (
              <>
                <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                AegisIRT Healing...
              </>
            ) : (
              <>
                <Shield className="h-5 w-5 mr-2" />
                Heal with AegisIRT AI
              </>
            )}
          </Button>
        </div>

        {/* Results */}
        {healingResults.length > 0 && (
          <div className="space-y-4">
            <h2 className="text-xl font-bold">AegisIRT Multi-Language Healing Results</h2>
            {healingResults.map((result) => (
              <Card key={result.id} className="border-l-4 border-l-slate-500">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                      {result.status === "approved" ? (
                        <CheckCircle className="h-5 w-5 text-green-600" />
                      ) : result.status === "rejected" ? (
                        <XCircle className="h-5 w-5 text-red-600" />
                      ) : result.status === "corrected" ? (
                        <Edit3 className="h-5 w-5 text-blue-600" />
                      ) : (
                        <AlertCircle className="h-5 w-5 text-yellow-600" />
                      )}
                      {result.element_name}
                    </CardTitle>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="p-6 bg-gradient-to-br from-red-50 to-red-100 rounded-xl border-2 border-red-200 shadow-sm">
                      <div className="flex items-center gap-2 mb-3">
                        <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                        <h4 className="font-bold text-red-800">❌ Broken Locator</h4>
                      </div>
                      <div className="bg-white p-3 rounded-lg border border-red-200">
                        <p className="text-sm text-red-700 font-mono break-all">
                          {result.original_locator.type} = "{result.original_locator.value}"
                        </p>
                      </div>
                    </div>

                    <div className="p-6 bg-gradient-to-br from-slate-50 to-blue-100 rounded-xl border-2 border-slate-200 shadow-sm">
                      <div className="flex items-center gap-2 mb-3">
                        <div className="w-3 h-3 bg-slate-600 rounded-full"></div>
                        <h4 className="font-bold text-slate-800">🛡️ AegisIRT Healed Locator</h4>
                      </div>
                      <div className="bg-white p-3 rounded-lg border border-slate-200">
                        <p className="text-sm text-slate-700 font-mono break-all">
                          {result.ai_suggestion.type} = "{result.ai_suggestion.value}"
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Language-Specific Code */}

                  <div>
                    <h4 className="font-semibold mb-2">AegisIRT Neural Network Confidence</h4>
                    <Progress value={result.ai_suggestion.confidence * 100} className="h-3" />
                    <p className="text-sm font-semibold mt-1">{(result.ai_suggestion.confidence * 100).toFixed(1)}%</p>
                  </div>

                  {result.status === "pending" && (
                    <div className="border-t pt-4">
                      <h4 className="font-semibold mb-3">🧠 Train AegisIRT Neural Network</h4>
                      <div className="flex gap-3">
                        <Button
                          onClick={() => provideFeedback(result.id, "approve")}
                          className="bg-green-600 hover:bg-green-700 text-white"
                        >
                          <ThumbsUp className="h-4 w-4 mr-2" />
                          AegisIRT Healed Correctly
                        </Button>
                        <Button onClick={() => provideFeedback(result.id, "reject")} variant="destructive">
                          <ThumbsDown className="h-4 w-4 mr-2" />
                          AegisIRT Healing Failed
                        </Button>
                        <Button
                          onClick={() => startCorrection(result)}
                          variant="outline"
                          className="border-slate-500 text-slate-600 hover:bg-slate-50"
                        >
                          <Edit3 className="h-4 w-4 mr-2" />
                          Correct & Train AegisIRT
                        </Button>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {/* Correction Modal */}
        {correctionMode && selectedResult && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <Card className="w-full max-w-2xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Edit3 className="h-5 w-5" />
                  Correct AegisIRT Healing & Train Neural Network
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="corrected-type">Correct Locator Type</Label>
                    <Select value={correctedType} onValueChange={setCorrectedType}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="By.ID">By.ID</SelectItem>
                        <SelectItem value="By.NAME">By.NAME</SelectItem>
                        <SelectItem value="By.CLASS_NAME">By.CLASS_NAME</SelectItem>
                        <SelectItem value="By.XPATH">By.XPATH</SelectItem>
                        <SelectItem value="By.CSS_SELECTOR">By.CSS_SELECTOR</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="corrected-value">Correct Locator Value</Label>
                    <Input
                      id="corrected-value"
                      value={correctedValue}
                      onChange={(e) => setCorrectedValue(e.target.value)}
                      placeholder="Enter correct locator value"
                    />
                  </div>
                </div>

                <div className="flex gap-3 justify-end">
                  <Button variant="outline" onClick={() => setCorrectionMode(false)}>
                    Cancel
                  </Button>
                  <Button
                    onClick={submitCorrection}
                    className="bg-slate-600 hover:bg-slate-700"
                    disabled={!correctedValue}
                  >
                    <Send className="h-4 w-4 mr-2" />
                    Train AegisIRT Neural Network
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
      <footer className="text-center text-gray-500 text-sm mt-8 pb-4">A K.V.S.K Production</footer>
    </div>
  )
}