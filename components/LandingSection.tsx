import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function LandingSection() {
  return (
    <div className="flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl text-center bg-white/90 backdrop-blur-sm shadow-2xl rounded-xl">
        <CardHeader>
          <CardTitle className="text-4xl font-extrabold text-gray-900 mb-4">Welcome to AegisIRT</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-lg text-gray-700 leading-relaxed">
            <strong>AegisIRT</strong> is an advanced AI-powered self-healing system designed for Interactive Voice Response Systems
            (IVRS) and Interactive Web Response Technologies (IRT) in clinical trials.
          </p>
          <p className="text-lg text-gray-700 leading-relaxed">
            It leverages deep learning to automatically identify and suggest fixes for broken UI locators and elements.
            Our system learns from your feedback to ensure reliable automation in clinical environments.
          </p>
          <Link href="#main-application">
            <Button className="mt-8 px-8 py-4 text-lg font-semibold bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-full shadow-lg hover:from-purple-700 hover:to-indigo-700 transition-all duration-300 ease-in-out transform hover:scale-105">
              Enter AegisIRT Application
            </Button>
          </Link>
        </CardContent>
      </Card>
    </div>
  )
}
