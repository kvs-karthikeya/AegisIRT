import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time

def test_adaptive_trial_full_workflow(driver):

    # ============================================================
    # STEP 1: Site coordinator login with MFA
    # ============================================================
    driver.find_element(By.ID, "site_coordinator_username").send_keys("coord_site042")
    driver.find_element(By.ID, "site_coordinator_password").send_keys("GCP@Site042!")
    driver.find_element(By.CLASS_NAME, "mfa-login-submit").click()
    
    WebDriverWait(driver, 15).until(
        EC.visibility_of_element_located((By.ID, "mfa_verification_panel"))
    )
    
    driver.find_element(By.ID, "mfa_code_input").send_keys("847291")
    driver.find_element(By.CLASS_NAME, "mfa-verify-btn").click()
    
    WebDriverWait(driver, 15).until(
        EC.visibility_of_element_located((By.ID, "irt_dashboard_main"))
    )

    # ============================================================
    # STEP 2: Navigate to protocol amendment notice
    # ============================================================
    driver.find_element(By.ID, "protocol_amendments_nav").click()
    
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CLASS_NAME, "amendments-list-container"))
    )
    
    driver.find_element(By.XPATH,
        "//tr[@data-amendment-id='AMD-2024-003']//button[@class='review-amendment-btn']"
    ).click()
    
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, "amendment_detail_panel"))
    )
    
    driver.find_element(By.ID, "acknowledge_amendment_btn").click()
    driver.find_element(By.CLASS_NAME, "amendment-acknowledgment-confirm").click()

    # ============================================================
    # STEP 3: Screen new patient with updated eligibility criteria
    # ============================================================
    driver.find_element(By.ID, "new_patient_screening_nav").click()
    
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, "patient_screening_form"))
    )
    
    driver.find_element(By.NAME, "patient_initials").send_keys("J.D.")
    driver.find_element(By.NAME, "patient_dob").send_keys("15/03/1968")
    driver.find_element(By.NAME, "screening_number").send_keys("SCR-2024-0445")
    
    # New eligibility criteria added by amendment
    driver.find_element(By.ID, "hoehn_yahr_scale_input").send_keys("2.5")
    driver.find_element(By.CLASS_NAME, "updrs-score-field").send_keys("42")
    driver.find_element(By.ID, "disease_duration_years").send_keys("4")
    
    driver.find_element(By.CLASS_NAME, "run-eligibility-check-btn").click()
    
    WebDriverWait(driver, 15).until(
        EC.visibility_of_element_located((By.CLASS_NAME, "eligibility-result-panel"))
    )
    
    eligibility = driver.find_element(By.CLASS_NAME, "eligibility-outcome-badge")
    assert eligibility.text == "ELIGIBLE"
    
    driver.find_element(By.ID, "confirm_screening_btn").click()

    # ============================================================
    # STEP 4: Stratified block randomization with new high-dose arm
    # ============================================================
    driver.find_element(By.ID, "randomization_module_nav").click()
    
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, "randomization_dashboard"))
    )
    
    driver.find_element(By.XPATH,
        "//div[@class='pending-randomizations-list']//tr[@data-screening-ref='SCR-2024-0445']//button[@class='initiate-randomization-btn']"
    ).click()
    
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, "stratified_randomization_modal"))
    )
    
    # Stratification factors added by protocol amendment
    stratification = Select(driver.find_element(By.NAME, "disease_severity_stratum"))
    stratification.select_by_value("moderate")
    
    site_stratum = Select(driver.find_element(By.NAME, "geographic_region_stratum"))
    site_stratum.select_by_value("europe_western")
    
    age_stratum = Select(driver.find_element(By.NAME, "age_group_stratum"))
    age_stratum.select_by_value("55_to_65")
    
    driver.find_element(By.ID, "execute_stratified_randomization_btn").click()
    
    WebDriverWait(driver, 15).until(
        EC.visibility_of_element_located((By.CLASS_NAME, "randomization-result-container"))
    )
    
    assigned_arm = driver.find_element(By.CLASS_NAME, "assigned-treatment-arm-display")
    assert assigned_arm.text in ["Arm A — Placebo", "Arm B — Low Dose 50mg", "Arm C — High Dose 100mg"]
    
    driver.find_element(By.ID, "confirm_randomization_result_btn").click()

    # ============================================================
    # STEP 5: Dispense stratification-matched drug kit
    # ============================================================
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, "kit_dispensing_module"))
    )
    
    driver.find_element(By.CLASS_NAME, "auto-select-matched-kit-btn").click()
    
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CLASS_NAME, "kit-selection-result-panel"))
    )
    
    kit_number = driver.find_element(By.CLASS_NAME, "selected-kit-number-display")
    assert kit_number.text.startswith("KIT-")
    
    driver.find_element(By.ID, "dispensing_verification_checkbox").click()
    driver.find_element(By.ID, "confirm_dispensing_btn").click()
    
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CLASS_NAME, "dispensing-success-confirmation"))
    )

    # ============================================================
    # STEP 6: Record baseline UPDRS assessments
    # ============================================================
    driver.find_element(By.ID, "baseline_assessments_nav").click()
    
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, "baseline_assessment_form"))
    )
    
    driver.find_element(By.NAME, "updrs_part1_score").send_keys("8")
    driver.find_element(By.NAME, "updrs_part2_score").send_keys("12")
    driver.find_element(By.NAME, "updrs_part3_score").send_keys("22")
    driver.find_element(By.NAME, "updrs_part4_score").send_keys("3")
    driver.find_element(By.ID, "tremor_assessment_score").send_keys("2")
    driver.find_element(By.CLASS_NAME, "rigidity-score-input").send_keys("3")
    
    driver.find_element(By.ID, "save_baseline_assessments_btn").click()
    
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CLASS_NAME, "assessment-saved-confirmation"))
    )

    # ============================================================
    # STEP 7: Trigger emergency unblinding procedure
    # ============================================================
    driver.find_element(By.ID, "emergency_procedures_nav").click()
    
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, "emergency_procedures_panel"))
    )
    
    driver.find_element(By.CLASS_NAME, "initiate-unblinding-btn").click()
    
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CLASS_NAME, "unblinding-authorization-modal"))
    )
    
    driver.find_element(By.NAME, "unblinding_reason").send_keys("Serious adverse event requiring immediate treatment decision")
    driver.find_element(By.NAME, "authorizing_physician").send_keys("Dr. Sarah Mitchell")
    driver.find_element(By.ID, "physician_authorization_code").send_keys("AUTH-2024-9921")
    
    driver.find_element(By.CLASS_NAME, "confirm-unblinding-submit-btn").click()
    
    WebDriverWait(driver, 15).until(
        EC.visibility_of_element_located((By.CLASS_NAME, "unblinding-result-panel"))
    )
    
    treatment_revealed = driver.find_element(By.CLASS_NAME, "revealed-treatment-display")
    assert treatment_revealed.text != ""
    
    driver.find_element(By.ID, "acknowledge_unblinding_btn").click()

    # ============================================================
    # STEP 8: Generate GCP audit trail report
    # ============================================================
    driver.find_element(By.ID, "reports_and_audit_nav").click()
    
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, "reports_dashboard"))
    )
    
    driver.find_element(By.CLASS_NAME, "generate-audit-trail-btn").click()
    
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CLASS_NAME, "audit-report-config-panel"))
    )
    
    date_from = driver.find_element(By.NAME, "audit_date_from")
    date_from.clear()
    date_from.send_keys("01/01/2024")
    
    date_to = driver.find_element(By.NAME, "audit_date_to")
    date_to.clear()
    date_to.send_keys("31/12/2024")
    
    scope = Select(driver.find_element(By.NAME, "audit_scope_selector"))
    scope.select_by_value("full_trial")
    
    driver.find_element(By.ID, "include_unblinding_events_checkbox").click()
    driver.find_element(By.ID, "include_protocol_deviations_checkbox").click()
    driver.find_element(By.ID, "generate_audit_report_btn").click()
    
    WebDriverWait(driver, 30).until(
        EC.visibility_of_element_located((By.CLASS_NAME, "audit-report-ready-banner"))
    )
    
    driver.find_element(By.ID, "download_audit_pdf_btn").click()
    driver.find_element(By.ID, "export_21cfr_compliant_format_btn").click()
