"""
Selenium Script to Download Encumbrance Certificate from TN RegINet Portal
Usage: python getEncumbranceCertificate.py --district <district> --sro <sro> --village <village> --survey <survey> --subdivision <subdivision> --output <path>
"""

import argparse
import time
import os
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('encumbrance_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EncumbranceCertificateDownloader:
    """Class to handle encumbrance certificate download from TN RegINet portal"""
    
    def __init__(self, output_path):
        self.output_path = output_path
        self.driver = None
        self.wait = None
        
        # Calculate dates
        self.end_date = datetime.now() - timedelta(days=1)  # Yesterday
        self.start_date = datetime.now() - timedelta(days=30*365)  # 30 years ago
        
    def find_field_by_multiple_ids(self, id_list):
        """Try to find a field by multiple possible IDs"""
        for field_id in id_list:
            try:
                element = self.wait.until(
                    EC.presence_of_element_located((By.ID, field_id))
                )
                logger.info(f"Found field with ID: {field_id}")
                return element
            except TimeoutException:
                continue
        
        # If not found by ID, try by name attribute
        for field_name in id_list:
            try:
                element = self.wait.until(
                    EC.presence_of_element_located((By.NAME, field_name))
                )
                logger.info(f"Found field with name: {field_name}")
                return element
            except TimeoutException:
                continue
        
        raise NoSuchElementException(f"Could not find field with any of these IDs: {id_list}")
    
    def setup_driver(self):
        """Initialize Chrome driver with download preferences"""
        chrome_options = Options()
        
        # Configure download settings
        prefs = {
            "download.default_directory": self.output_path,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True  # Force PDF download instead of opening
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Optional: Run in headless mode (uncomment if needed)
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, 20)
            logger.info("Chrome driver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Chrome driver: {str(e)}")
            raise
    
    def navigate_to_portal(self):
        """Navigate to TN RegINet portal"""
        try:
            url = "https://tnreginet.gov.in/portal/"
            logger.info(f"Navigating to {url}")
            self.driver.get(url)
            time.sleep(3)  # Wait for page to load
            logger.info("Successfully loaded portal homepage")
            
            # Try to switch to English if portal is in Tamil
            self.switch_to_english()
            
        except Exception as e:
            logger.error(f"Failed to navigate to portal: {str(e)}")
            raise
    
    def switch_to_english(self):
        """Switch portal language to English if it's in Tamil"""
        try:
            logger.info("Checking portal language and switching to English if needed")
            
            # Look for language switch links/buttons
            english_selectors = [
                "//a[contains(text(), 'English')]",
                "//a[contains(text(), 'ENGLISH')]",
                "//button[contains(text(), 'English')]",
                "//a[@title='English']",
                "//a[contains(@href, 'lang=en') or contains(@href, 'language=en')]",
                "//select[@id='language']//option[@value='en']",
                "//div[contains(@class, 'language')]//a[contains(text(), 'En')]"
            ]
            
            for selector in english_selectors:
                try:
                    english_link = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    logger.info(f"Found language switcher, clicking to switch to English")
                    english_link.click()
                    time.sleep(2)
                    logger.info("Successfully switched to English")
                    return
                except TimeoutException:
                    continue
            
            logger.info("No language switcher found or already in English")
            
        except Exception as e:
            logger.warning(f"Could not switch language (may already be in English): {str(e)}")
    
    def navigate_to_encumbrance_section(self):
        """Navigate through menus: Electronic services -> Villangan Evidence -> Viewing details"""
        try:
            # Click on "Electronic Services" or "இ-சேவைகள்"
            logger.info("Looking for Electronic Services menu")
            
            # Try multiple possible selectors for the menu (English and Tamil)
            electronic_services_selectors = [
                "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'electronic services')]",
                "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'e-services')]",
                "//a[contains(text(), 'Electronic Services')]",
                "//a[contains(text(), 'இ-சேவைகள்')]",
                "//a[contains(@href, 'eservices') or contains(@href, 'electronic')]",
                "//li[contains(@class, 'menu')]//a[contains(text(), 'Services')]",
                "//a[contains(text(), 'E-Services')]"
            ]
            
            electronic_services = None
            for selector in electronic_services_selectors:
                try:
                    electronic_services = self.wait.until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    break
                except TimeoutException:
                    continue
            
            if not electronic_services:
                raise Exception("Could not find Electronic Services menu")
            
            electronic_services.click()
            logger.info("Clicked Electronic Services menu")
            time.sleep(2)
            
            # Click on "Villangan Evidence" / "வில்லங்கச் சான்று"
            logger.info("Looking for Villangan Evidence menu")
            villangan_selectors = [
                "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'encumbrance')]",
                "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'villangan')]",
                "//a[contains(text(), 'Villangan Evidence')]",
                "//a[contains(text(), 'வில்லங்கச் சான்று')]",
                "//a[contains(text(), 'Encumbrance Certificate')]",
                "//a[contains(text(), 'Encumbrance')]",
                "//a[contains(@href, 'villangan') or contains(@href, 'encumbrance')]"
            ]
            
            villangan_menu = None
            for selector in villangan_selectors:
                try:
                    villangan_menu = self.wait.until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    break
                except TimeoutException:
                    continue
            
            if not villangan_menu:
                raise Exception("Could not find Villangan Evidence menu")
            
            villangan_menu.click()
            logger.info("Clicked Villangan Evidence menu")
            time.sleep(2)
            
            # Click on "Viewing the details of the Villangan certificate"
            logger.info("Looking for Viewing Details option")
            viewing_selectors = [
                "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'view')]",
                "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'details')]",
                "//a[contains(text(), 'Viewing the details')]",
                "//a[contains(text(), 'வில்லங்க சான்றிதழின் விவரங்களைப் பார்த்தல்')]",
                "//a[contains(text(), 'View Certificate')]",
                "//a[contains(text(), 'View Details')]",
                "//a[contains(@href, 'view') or contains(@href, 'details')]"
            ]
            
            viewing_option = None
            for selector in viewing_selectors:
                try:
                    viewing_option = self.wait.until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    break
                except TimeoutException:
                    continue
            
            if not viewing_option:
                raise Exception("Could not find Viewing Details option")
            
            viewing_option.click()
            logger.info("Clicked Viewing Details option")
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"Failed to navigate to encumbrance section: {str(e)}")
            logger.info("Taking screenshot for debugging")
            self.driver.save_screenshot(os.path.join(self.output_path, "navigation_error.png"))
            raise
    
    def fill_property_details(self, zone, district, sro, village, survey_number, subdivision):
        """Fill in the property details form"""
        try:
            logger.info("Filling property details form")
            
            # Select Zone
            logger.info(f"Selecting Zone: {zone}")
            zone_field = self.find_field_by_multiple_ids([
                "zone", "Zone", "zoneId", "zoneName", "மண்டலம்"
            ])
            zone_select = Select(zone_field)
            zone_select.select_by_visible_text(zone)
            time.sleep(1)
            
            # Select District
            logger.info(f"Selecting District: {district}")
            district_field = self.find_field_by_multiple_ids([
                "district", "District", "districtId", "districtName", "மாவட்டம்"
            ])
            district_select = Select(district_field)
            district_select.select_by_visible_text(district)
            time.sleep(1)
            
            # Select Sub-Registrar's Office
            logger.info(f"Selecting SRO: {sro}")
            sro_field = self.find_field_by_multiple_ids([
                "sro", "SRO", "sroId", "sroName", "subRegistrar", "துணை பதிவாளர்"
            ])
            sro_select = Select(sro_field)
            sro_select.select_by_visible_text(sro)
            time.sleep(1)
            
            # Select Registration Village
            logger.info(f"Selecting Village: {village}")
            village_field = self.find_field_by_multiple_ids([
                "village", "Village", "villageId", "villageName", "registrationVillage", "பதிவு கிராமம்"
            ])
            village_select = Select(village_field)
            village_select.select_by_visible_text(village)
            time.sleep(1)
            
            # Fill Start Date (30 years ago)
            logger.info(f"Entering Start Date: {self.start_date.strftime('%d/%m/%Y')}")
            start_date_field = self.find_field_by_multiple_ids([
                "startDate", "fromDate", "dateFrom", "start_date", "தொடக்க தேதி"
            ])
            start_date_field.clear()
            start_date_field.send_keys(self.start_date.strftime("%d/%m/%Y"))
            
            # Fill End Date (Yesterday)
            logger.info(f"Entering End Date: {self.end_date.strftime('%d/%m/%Y')}")
            end_date_field = self.find_field_by_multiple_ids([
                "endDate", "toDate", "dateTo", "end_date", "முடிவு தேதி"
            ])
            end_date_field.clear()
            end_date_field.send_keys(self.end_date.strftime("%d/%m/%Y"))
            
            # Fill Survey Number (Field Number)
            logger.info(f"Entering Survey Number: {survey_number}")
            survey_field = self.find_field_by_multiple_ids([
                "surveyNumber", "fieldNumber", "survey_number", "field_number", "புல எண்"
            ])
            survey_field.clear()
            survey_field.send_keys(survey_number)
            
            # Fill Subdivision if provided
            if subdivision:
                logger.info(f"Entering Subdivision: {subdivision}")
                try:
                    subdivision_field = self.find_field_by_multiple_ids([
                        "subdivision", "subDivision", "sub_division"
                    ])
                    subdivision_field.clear()
                    subdivision_field.send_keys(subdivision)
                except NoSuchElementException:
                    logger.warning("Subdivision field not found, skipping")
            
            logger.info("Property details filled successfully")
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Failed to fill property details: {str(e)}")
            self.driver.save_screenshot(os.path.join(self.output_path, "form_fill_error.png"))
            raise
    
    def submit_and_download(self):
        """Click ADD button and handle download"""
        try:
            # Click ADD button
            logger.info("Looking for ADD button")
            add_button_selectors = [
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'add')]",
                "//button[contains(text(), 'ADD')]",
                "//button[contains(text(), 'Add')]",
                "//button[contains(text(), 'சேர்')]",
                "//input[@type='button' and contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'add')]",
                "//input[@type='submit' and contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'add')]",
                "//button[@id='addButton' or @id='btnAdd' or @id='add']",
                "//input[@type='submit']"
            ]
            
            add_button = None
            for selector in add_button_selectors:
                try:
                    add_button = self.wait.until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    break
                except TimeoutException:
                    continue
            
            if not add_button:
                raise Exception("Could not find ADD button")
            
            logger.info("Clicking ADD button")
            add_button.click()
            time.sleep(5)
            
            # Wait for download or search results
            logger.info("Waiting for search results or download to start")
            
            # Check if download button appears
            download_button_selectors = [
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download')]",
                "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download')]",
                "//button[contains(text(), 'Download')]",
                "//a[contains(text(), 'Download')]",
                "//button[contains(text(), 'பதிவிறக்க')]",
                "//button[contains(text(), 'Print')]",
                "//button[contains(text(), 'அச்சிடு')]",
                "//a[contains(@href, 'download') or contains(@href, 'pdf')]"
            ]
            
            download_button = None
            for selector in download_button_selectors:
                try:
                    download_button = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    break
                except TimeoutException:
                    continue
            
            if download_button:
                logger.info("Found download button, clicking to download certificate")
                download_button.click()
                time.sleep(10)  # Wait for download to complete
            else:
                logger.warning("No explicit download button found, checking if auto-download occurred")
                time.sleep(10)
            
            # Verify download
            self.verify_download()
            
        except Exception as e:
            logger.error(f"Failed to submit and download: {str(e)}")
            self.driver.save_screenshot(os.path.join(self.output_path, "submit_error.png"))
            raise
    
    def verify_download(self):
        """Verify that the certificate was downloaded"""
        try:
            # Wait a bit more for download to complete
            time.sleep(5)
            
            # Check for PDF files in download directory
            files = os.listdir(self.output_path)
            pdf_files = [f for f in files if f.lower().endswith('.pdf')]
            
            if pdf_files:
                logger.info(f"Download successful! Found {len(pdf_files)} PDF file(s)")
                for pdf in pdf_files:
                    logger.info(f"  - {pdf}")
                return True
            else:
                logger.warning("No PDF files found in download directory")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying download: {str(e)}")
            return False
    
    def close(self):
        """Close the browser"""
        if self.driver:
            logger.info("Closing browser")
            self.driver.quit()
    
    def download_certificate(self, zone, district, sro, village, survey_number, subdivision=""):
        """Main method to download encumbrance certificate"""
        try:
            self.setup_driver()
            self.navigate_to_portal()
            self.navigate_to_encumbrance_section()
            self.fill_property_details(zone, district, sro, village, survey_number, subdivision)
            self.submit_and_download()
            logger.info("Certificate download process completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Certificate download failed: {str(e)}")
            return False
            
        finally:
            self.close()


def main():
    """Main function to parse arguments and execute download"""
    parser = argparse.ArgumentParser(
        description='Download Encumbrance Certificate from TN RegINet Portal'
    )
    parser.add_argument('--district', required=True, help='District name')
    parser.add_argument('--sro', required=True, help='Sub-Registrar Office name')
    parser.add_argument('--village', required=True, help='Village name')
    parser.add_argument('--survey', required=True, help='Survey number')
    parser.add_argument('--subdivision', default='', help='Subdivision (optional)')
    parser.add_argument('--output', required=True, help='Output directory path')
    parser.add_argument('--zone', default='Chennai', help='Zone name (default: Chennai)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
        logger.info(f"Created output directory: {args.output}")
    
    # Log the parameters
    logger.info("="*60)
    logger.info("Starting Encumbrance Certificate Download")
    logger.info("="*60)
    logger.info(f"Zone: {args.zone}")
    logger.info(f"District: {args.district}")
    logger.info(f"SRO: {args.sro}")
    logger.info(f"Village: {args.village}")
    logger.info(f"Survey Number: {args.survey}")
    logger.info(f"Subdivision: {args.subdivision if args.subdivision else 'N/A'}")
    logger.info(f"Output Path: {args.output}")
    logger.info("="*60)
    
    # Create downloader instance and execute
    downloader = EncumbranceCertificateDownloader(args.output)
    success = downloader.download_certificate(
        zone=args.zone,
        district=args.district,
        sro=args.sro,
        village=args.village,
        survey_number=args.survey,
        subdivision=args.subdivision
    )
    
    if success:
        logger.info("✓ Process completed successfully")
        exit(0)
    else:
        logger.error("✗ Process failed")
        exit(1)


if __name__ == "__main__":
    main()
