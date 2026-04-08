import re
import json

test_code = open('test_input.py').read()
dom_html = open('test_dom.html').read()

keywords = ['locator', 'getByTestId', 'getByText', 'getByRole', 'getByLabel', 'findElement', 'find_element', 'FindElement']
locators = []
for kw in keywords:
    idx = 0
    while True:
        idx = test_code.find(kw, idx)
        if idx == -1: break
        
        open_paren = test_code.find('(', idx)
        if open_paren != -1 and open_paren < idx + len(kw) + 2:
            paren_count = 1
            curr = open_paren + 1
            in_string = False
            string_char = ''
            
            while curr < len(test_code) and paren_count > 0:
                char = test_code[curr]
                if in_string:
                    if char == string_char and test_code[curr-1] != '\\':
                        in_string = False
                else:
                    if char in ["'", '"', '`']:
                        in_string = True
                        string_char = char
                    elif char == '(': paren_count += 1
                    elif char == ')': paren_count -= 1
                curr += 1
            
            if paren_count == 0:
                snippet = test_code[idx:curr]
                locators.append({'snippet': snippet, 'idx': idx})
        idx += len(kw)

locators.sort(key=lambda x: x['idx'])
print("Total locators found:", len(locators))

# We will manually mock the choices
assignments = [
    # 1. Site coordinator login
    ("[data-testid='coordinator-username-field']", "CSS_SELECTOR", "found using new data-testid auth pattern"),
    ("[data-testid='coordinator-password-field']", "CSS_SELECTOR", "matched field structure"),
    ("[data-testid='submit-login-mfa']", "CSS_SELECTOR", "found mfa login submission button"),
    # mfa verify panel wait
    ("[data-testid='mfa-verification-panel']", "CSS_SELECTOR", "matched new panel id"),
    ("[data-testid='mfa-code-entry']", "CSS_SELECTOR", "matched mfa code entry"),
    ("[data-testid='submit-mfa-verification']", "CSS_SELECTOR", "matched verify button"),
    # dashboard main wait
    ("[data-testid='irt-main-dashboard']", "CSS_SELECTOR", "dashboard wrapper"),
    
    # 2. Navigate protocol amendment
    ("[data-testid='nav-protocol-amendments']", "CSS_SELECTOR", "nav button"),
    (".amendments-list", "CSS_SELECTOR", "list container"), # wait class changes to amendments-list
    ("[data-testid='review-amendment-action'][data-amendment-ref='AMD-2024-003']", "CSS_SELECTOR", "row review button"),
    ("[data-testid='amendment-detail-view']", "CSS_SELECTOR", "panel"),
    ("[data-testid='acknowledge-amendment-btn']", "CSS_SELECTOR", "acknowledge button"),
    ("[data-testid='confirm-acknowledgment-action']", "CSS_SELECTOR", "confirm button"),
    
    # 3. Patient Screening
    ("[data-testid='nav-patient-screening']", "CSS_SELECTOR", "nav button"),
    ("[data-testid='patient-screening-form']", "CSS_SELECTOR", "form"),
    ("[data-testid='patient-initials-input']", "CSS_SELECTOR", "initials field"),
    ("[data-testid='patient-dob-input']", "CSS_SELECTOR", "dob field"),
    ("[data-testid='screening-number-input']", "CSS_SELECTOR", "screening number field"),
    ("[data-testid='hoehn-yahr-input']", "CSS_SELECTOR", "hoehn yahr field"),
    ("[data-testid='updrs-total-score-input']", "CSS_SELECTOR", "updrs score field"),
    ("[data-testid='disease-duration-input']", "CSS_SELECTOR", "disease duration"),
    ("[data-testid='execute-eligibility-check']", "CSS_SELECTOR", "check button"),
    ("[data-testid='eligibility-result-wrapper']", "CSS_SELECTOR", "result panel"),
    ("[data-testid='eligibility-outcome-indicator']", "CSS_SELECTOR", "outcome badge"),
    ("[data-testid='confirm-screening-submission']", "CSS_SELECTOR", "confirm button"),
    
    # 4. Randomization
    ("[data-testid='nav-randomization']", "CSS_SELECTOR", "randomization nav"),
    ("[data-testid='randomization-dashboard-view']", "CSS_SELECTOR", "dashboard"),
    ("[data-testid='begin-randomization-action'][data-screening-ref='SCR-2024-0445']", "CSS_SELECTOR", "initiate rand"),
    ("[data-testid='stratified-randomization-dialog']", "CSS_SELECTOR", "modal"),
    ("[data-testid='disease-severity-stratum-select']", "CSS_SELECTOR", "severity select"),
    ("[data-testid='geographic-region-stratum-select']", "CSS_SELECTOR", "region select"),
    ("[data-testid='age-group-stratum-select']", "CSS_SELECTOR", "age select"),
    ("[data-testid='submit-stratified-randomization']", "CSS_SELECTOR", "execute rand"),
    ("[data-testid='randomization-outcome-panel']", "CSS_SELECTOR", "outcome panel"),
    ("[data-testid='treatment-arm-assignment-display']", "CSS_SELECTOR", "assignment display"),
    ("[data-testid='confirm-randomization-outcome']", "CSS_SELECTOR", "confirm result"),
    
    # 5. Kit Dispensing
    ("[data-testid='kit-dispensing-module']", "CSS_SELECTOR", "dispensing module"),
    ("[data-testid='auto-select-kit-action']", "CSS_SELECTOR", "auto select matched"),
    ("[data-testid='kit-selection-result']", "CSS_SELECTOR", "selection result"),
    ("[data-testid='selected-kit-number']", "CSS_SELECTOR", "kit number display"),
    ("[data-testid='dispensing-verification-checkbox']", "CSS_SELECTOR", "verification checkbox"),
    ("[data-testid='submit-dispensing-confirmation']", "CSS_SELECTOR", "confirm dispensing"),
    ("[data-testid='dispensing-success-banner']", "CSS_SELECTOR", "success banner"),
    
    # 6. Baseline Assessments
    ("[data-testid='nav-baseline-assessments']", "CSS_SELECTOR", "nav baseline"),
    ("[data-testid='baseline-assessment-form']", "CSS_SELECTOR", "form"),
    ("[data-testid='updrs-part1-input']", "CSS_SELECTOR", "part 1"),
    ("[data-testid='updrs-part2-input']", "CSS_SELECTOR", "part 2"),
    ("[data-testid='updrs-part3-input']", "CSS_SELECTOR", "part 3"),
    ("[data-testid='updrs-part4-input']", "CSS_SELECTOR", "part 4"),
    ("[data-testid='tremor-assessment-input']", "CSS_SELECTOR", "tremor input"),
    ("[data-testid='rigidity-score-input']", "CSS_SELECTOR", "rigidity input"),
    ("[data-testid='save-baseline-data']", "CSS_SELECTOR", "save button"),
    ("[data-testid='assessment-save-confirmation']", "CSS_SELECTOR", "save confirmation"),
    
    # 7. Unblinding
    ("[data-testid='nav-emergency-procedures']", "CSS_SELECTOR", "emergency nav"),
    ("[data-testid='emergency-procedures-panel']", "ID", "doesn't strictly exist, map to module"), # mapped below
    ("[data-testid='initiate-emergency-unblinding']", "CSS_SELECTOR", "initiate"),
    ("[data-testid='unblinding-authorization-dialog']", "CSS_SELECTOR", "dialog"),
    ("[data-testid='unblinding-reason-input']", "CSS_SELECTOR", "reason"),
    ("[data-testid='authorizing-physician-input']", "CSS_SELECTOR", "physician"),
    ("[data-testid='physician-auth-code-input']", "CSS_SELECTOR", "code"),
    ("[data-testid='submit-unblinding-request']", "CSS_SELECTOR", "submit"),
    ("[data-testid='unblinding-result-container']", "CSS_SELECTOR", "result container"),
    ("[data-testid='revealed-treatment-assignment']", "CSS_SELECTOR", "revealed"),
    ("[data-testid='acknowledge-unblinding-result']", "CSS_SELECTOR", "acknowledge"),
    
    # 8. Reports
    ("[data-testid='nav-reports-audit']", "CSS_SELECTOR", "reports nav"),
    ("[data-testid='reports-dashboard-view']", "CSS_SELECTOR", "reports dashboard"),
    ("[data-testid='generate-audit-trail-action']", "CSS_SELECTOR", "generate action"),
    ("[data-testid='audit-report-configuration']", "CSS_SELECTOR", "config panel"),
    ("[data-testid='audit-date-from-input']", "CSS_SELECTOR", "date from"),
    ("[data-testid='audit-date-to-input']", "CSS_SELECTOR", "date to"),
    ("[data-testid='audit-scope-select']", "CSS_SELECTOR", "scope"),
    ("[data-testid='include-unblinding-checkbox']", "CSS_SELECTOR", "include unblinding"),
    ("[data-testid='include-deviations-checkbox']", "CSS_SELECTOR", "include deviations"),
    ("[data-testid='execute-audit-report-generation']", "CSS_SELECTOR", "generate"),
    ("[data-testid='audit-report-ready-notification']", "CSS_SELECTOR", "ready"),
    ("[data-testid='download-audit-pdf-action']", "CSS_SELECTOR", "download"),
    ("[data-testid='export-cfr-compliant-format']", "CSS_SELECTOR", "export")
]

# Adjust mapping for panels that don't uniquely match
panel_overrides = {
    "emergency_procedures_panel": "[data-module='emergency-procedures']"
}

output = []
for i, (val, typ, reason) in enumerate(assignments):
    if i < len(locators):
        orig = locators[i]['snippet']
        if "emergency_procedures_panel" in orig:
            val = "[data-module='emergency-procedures']"
        output.append({
            "id": i,
            "type": typ,
            "value": val,
            "confidence": 0.98,
            "reasoning": f"Testing output auto-mapping: {reason}"
        })

print(json.dumps(output, indent=2))
with open('c:/Users/Sai/OneDrive/Desktop/aegisIRT/mock_gemini.json', 'w') as f:
    json.dump(output, f)
