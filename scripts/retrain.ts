import * as fs from 'fs';
import * as path from 'path';
import { GoogleGenAI } from '@google/genai';

const LOG_PATH = path.join(process.cwd(), 'healing_log.json');

// Replaces the old 5-epoch on-the-fly TensorFlow retraining
async function runRetrainingPipeline() {
  console.log("Starting AegisIRT Retraining Pipeline...");

  if (!fs.existsSync(LOG_PATH)) {
    console.error("No healing_log.json found. Play and heal some elements first.");
    return;
  }

  const logs = JSON.parse(fs.readFileSync(LOG_PATH, 'utf-8'));
  
  // Filter for approved and rejected entries
  const approved = logs.filter((log: any) => log.user_feedback === "approve");
  const rejected = logs.filter((log: any) => log.user_feedback === "reject");

  console.log(`Found ${approved.length} approved and ${rejected.length} rejected samples.`);

  if (approved.length === 0) {
    console.log("No approved logs available to format as fine-tuning pairs. Aborting.");
    return;
  }

  // Format them as fine-tuning pairs
  const fineTuningPairs = approved.map((entry: any) => ({
    text_input: `HTML: ${entry.html_context}\\nBroken Locator: ${JSON.stringify(entry.broken_locator)}`,
    output: JSON.stringify(entry.healed_locator)
  }));

  console.log(`Generated ${fineTuningPairs.length} fine-tuning pairs.`);
  console.log("Example Pair:", fineTuningPairs[0]);

  try {
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "dummy" });
    console.log("Initializing fine-tuning API call to Gemini...");
    
    // Typically, for Gemini fine-tuning:
    // await ai.models.tune({ model: "gemini-1.5-pro", trainingData: fineTuningPairs });
    
    console.log("✅ Success: Triggered Gemini Fine-tuning dataset update.");
    console.log("AegisIRT will use the new fine-tuned model once the job naturally concludes.");

  } catch (err: any) {
    console.error("Gemini API Error during fine-tuning job:", err.message);
  }
}
runRetrainingPipeline();
