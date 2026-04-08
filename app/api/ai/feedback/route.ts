import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { prediction_id, feedback_type, corrected_solution, notes } = body

    // Send feedback to your actual AI model for retraining
    const feedbackResponse = await fetch("http://localhost:5000/feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prediction_id,
        feedback_type, // 'approve', 'reject', or 'correct'
        corrected_solution,
        notes,
      }),
    })

    if (!feedbackResponse.ok) {
      throw new Error("Failed to submit feedback to AI model")
    }

    const result = await feedbackResponse.json()
    return NextResponse.json(result)
  } catch (error) {
    console.error("Feedback submission error:", error)
    return NextResponse.json({ error: "Failed to submit feedback" }, { status: 500 })
  }
}
