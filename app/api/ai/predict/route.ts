import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { element_info, html_content, clinical_context } = body

    // This would connect to your actual Python AI model
    // For now, we'll call a mock endpoint that represents your real model
    const aiResponse = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        element_info,
        html_content,
        clinical_context,
      }),
    })

    if (!aiResponse.ok) {
      throw new Error("AI model prediction failed")
    }

    const prediction = await aiResponse.json()
    return NextResponse.json(prediction)
  } catch (error) {
    console.error("AI prediction error:", error)
    return NextResponse.json({ error: "Failed to get AI prediction" }, { status: 500 })
  }
}
