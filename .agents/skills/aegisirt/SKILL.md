---
name: "AegisIRT Clinical Trials"
description: "Codifies AegisIRT's clinical trial domain knowledge — IRT workflows, patient enrollment, randomization, drug dispensing — so the agent follows it without being re-prompted."
---

# AegisIRT Clinical Trial Domain Knowledge

When working on AegisIRT, use the following clinical trial domain knowledge:

## 1. IRT (Interactive Response Technology) Workflows
IRT systems manage patient interactions, logistics, and data in clinical trials.
- Ensure strict audit trails for all actions.
- Double-blind testing paradigms must be maintained.
- Prioritize data integrity and validation of user roles (e.g., Investigators, CRAs, Sponsors).

## 2. Patient Enrollment
The process of registering a subject into the clinical trial.
- Key elements: `Subject ID`, `Inclusion/Exclusion criteria checklist`, `Informed Consent tracking`.
- Common UI interactions: "Enroll Patient", "Screen Subject", "Save Demographics".

## 3. Randomization
Assigning enrolled patients to specific treatment arms based on trial protocol.
- Must handle stratification factors (e.g., age, disease severity).
- Common UI interactions: "Randomize Subject", "Assign Treatment Arm".
- Note: The actual allocation arm must typically remain blinded to the site user.

## 4. Drug Dispensing (Study Medication)
Allocating and tracking the actual investigational product to the subject.
- Key elements: `Kit numbers`, `Lot numbers`, `Dispensation schedules (visits)`.
- Common UI interactions: "Dispense Medication", "Assign Kit", "Record Returns", "Mark as Damaged/Lost".

## Design & Aesthetic Rules (AegisIRT specific)
- Apply a black background `#080808` with white text and white borders throughout. 
- Cards use `#0e0e0e` background with `#1e1e1e` borders. 
- Active and highlighted elements use pure white `#ffffff`. 
- Muted text is `#555`. 
- Code blocks use `#111` background. 
- No gradients, no colors except white accents.
