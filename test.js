const fs = require('fs');

async function run() {
  const testCode = fs.readFileSync('test_input.py', 'utf8');
  const domHtml = fs.readFileSync('test_dom.html', 'utf8');

  console.log("Sending payload to API...");
  try {
    const res = await fetch('http://localhost:3000/api/ai', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: 'heal_batch',
        test_code: testCode,
        html_content: domHtml,
        clinical_context: 'Test'
      })
    });

    const json = await res.json();
    console.log("API Response Status:", res.status);
    
    if (json.healed_test_code) {
        fs.writeFileSync('test_output.py', json.healed_test_code);
        console.log(`Successfully healed code. Wrote to test_output.py. Total locators healed: ${json.results.length}`);
    } else {
        console.log("Failed to heal.", json);
    }
  } catch (err) {
      console.error("Error hitting Next.js server. Is it running on port 3000?", err.message);
  }
}
run();
