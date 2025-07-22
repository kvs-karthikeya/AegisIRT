import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  try {
    // Get real model status from your AI backend
    const statusResponse = await fetch("http://localhost:5000/status", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!statusResponse.ok) {
      throw new Error("Failed to get model status")
    }

    const status = await statusResponse.json()
    return NextResponse.json(status)
  } catch (error) {
    console.error("Status check error:", error)
    return NextResponse.json({ error: "Failed to get model status" }, { status: 500 })
  }
}
