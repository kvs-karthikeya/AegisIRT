// components/LandingSection.tsx
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface LandingProps {
  onEnter: () => void
}

export default function LandingSection({ onEnter }: LandingProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 to-gray-700 p-4">
      <Card className="w-full max-w-2xl text-center bg-white/90 backdrop-blur-sm shadow-2xl rounded-xl">
        <CardHeader>
          <CardTitle className="text-4xl font-extrabold text-gray-900 mb-4">Welcome to AegisIRT</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-lg text-gray-700 leading-relaxed">
            <strong>AegisIRT</strong> is a self-healing AI backend for IVRS/IRT systems in clinical trials. It automatically repairs broken UI locators and streamlines critical workflows.
          </p>
          <Button
            onClick={onEnter}
            className="mt-8 px-8 py-4 text-lg font-semibold bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-full shadow-lg hover:from-purple-700 hover:to-indigo-700 transition-all duration-300 ease-in-out transform hover:scale-105"
          >
            Enter AegisIRT Application
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}

