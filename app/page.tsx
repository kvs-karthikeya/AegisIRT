import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function LandingPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 to-gray-700 p-4">
      <Card className="w-full max-w-2xl text-center bg-white/90 backdrop-blur-sm shadow-2xl rounded-xl">
        <CardHeader>
          <CardTitle className="text-4xl font-extrabold text-gray-900 mb-4">Welcome to AegisIRT</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-lg text-gray-700 leading-relaxed">
            **AegisIRT** is an advanced AI-powered self-healing system designed for Interactive Voice Response Systems
            (IVRS) and Interactive Web Response Technologies (IRT) in clinical trials. It leverages deep learning to
            automatically identify and suggest fixes for broken UI locators and elements across various programming
            languages.
          </p>
          <p className="text-lg text-gray-700 leading-relaxed">
            Our system continuously learns from your feedback, ensuring robust and reliable automation in complex
            clinical trial environments.
          </p>
          <Link href="/main" passHref>
            <Button className="mt-8 px-8 py-4 text-lg font-semibold bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-full shadow-lg hover:from-purple-700 hover:to-indigo-700 transition-all duration-300 ease-in-out transform hover:scale-105">
              Enter AegisIRT Application
            </Button>
          </Link>
        </CardContent>
      </Card>
    </div>
  )
}
