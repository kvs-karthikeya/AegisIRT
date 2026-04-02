import { Shield } from "lucide-react"

interface LandingSectionProps {
  onEnter: () => void;
}

export default function LandingSection({ onEnter }: LandingSectionProps) {
  return (
    <div className="min-h-screen bg-[#080808] text-white flex flex-col items-center justify-center font-sans tracking-wide" style={{ fontFamily: '"Open Sans", sans-serif' }}>
      <div className="max-w-4xl mx-auto px-6 text-center flex flex-col items-center">
        <Shield className="w-20 h-20 text-white mb-8" />
        
        <h1 className="text-4xl md:text-5xl font-bold text-white mb-6">
          Stop fixing broken IRT test locators manually
        </h1>
        
        <p className="text-base md:text-lg text-white mb-10 max-w-2xl leading-relaxed">
          AegisIRT uses Gemini AI and browser verification to heal broken Selenium<br className="hidden md:block"/>
          and Playwright locators automatically for clinical trial systems.
        </p>

        <div className="flex flex-wrap justify-center gap-4 mb-14">
          <div className="border border-white bg-black text-white px-6 py-2 rounded-full text-sm font-medium">Smart Parsing</div>
          <div className="border border-white bg-black text-white px-6 py-2 rounded-full text-sm font-medium">Gemini AI Healing</div>
          <div className="border border-white bg-black text-white px-6 py-2 rounded-full text-sm font-medium">Browser Verified</div>
          <div className="border border-white bg-black text-white px-6 py-2 rounded-full text-sm font-medium">Learns from Feedback</div>
        </div>

        <button 
          onClick={onEnter}
          className="bg-white text-black hover:bg-gray-200 border-2 border-white transition-all font-bold px-12 py-4 rounded-none tracking-widest uppercase text-sm"
        >
          Enter
        </button>
      </div>
    </div>
  )
}
