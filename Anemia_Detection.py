import cv2
import numpy as np
import json
from datetime import datetime
from scipy.signal import savgol_filter
import os

def load_specific_images(nail_path='Anemia_img/nail_bed.jpg', eyelid_path='Anemia_img/eyelid.jpg'):
    """Load nail bed and conjunctiva/eyelid images specifically"""
    nail_bed_img = cv2.imread(nail_path)  # Nail bed close-up
    conjunctiva_img = cv2.imread(eyelid_path)  # Conjunctiva close-up (technically not eyelid)
    
    if nail_bed_img is None or conjunctiva_img is None:
        raise FileNotFoundError(f"Could not load images. Please ensure you have '{os.path.basename(nail_path)}' and '{os.path.basename(eyelid_path)}' in the correct folder.")
    
    return nail_bed_img, conjunctiva_img

def analyze_image_quality(image):
    """Analyze image for multiple quality metrics"""
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate sharpness using Laplacian variance
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate brightness (mean intensity)
    brightness = np.mean(gray)
    
    # Calculate contrast (standard deviation of intensity)
    contrast = np.std(gray)
    
    # Check for motion blur
    motion_blur = cv2.blur(gray, (5, 5))
    blur_difference = np.abs(gray.astype(np.float32) - motion_blur.astype(np.float32))
    blur_score = np.mean(blur_difference)
    
    return {
        'sharpness': float(sharpness),
        'brightness': float(brightness),
        'contrast': float(contrast),
        'blur_score': float(blur_score)
    }

def enhance_image_for_medical_analysis(image):
    """Medical-grade image enhancement specifically tuned for pallor detection
    
    This preserves color information critical for anemia assessment while
    reducing noise and improving contrast
    """
    # IMPORTANT: For anemia detection, we need to be very careful with color enhancement
    # as it can mask or exaggerate pallor. Use more conservative enhancements.
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply gentle CLAHE to luminance channel (adaptive enhancement)
    # Using smaller clip limit to avoid over-enhancing pink tones
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Reconstruct image
    enhanced_lab = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Minimal denoising (preserves color information important for pallor detection)
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 7, 7, 5, 15)
    
    # Print debugging info about color preservation
    bgr_b, bgr_g, bgr_r = cv2.split(image)
    enh_b, enh_g, enh_r = cv2.split(denoised)
    
    print("Color preservation check:")
    print(f"  Original - R:{np.mean(bgr_r):.1f}, G:{np.mean(bgr_g):.1f}, B:{np.mean(bgr_b):.1f}")
    print(f"  Enhanced - R:{np.mean(enh_r):.1f}, G:{np.mean(enh_g):.1f}, B:{np.mean(enh_b):.1f}")
    
    return denoised

def extract_nail_bed_roi(image):
    """Extract nail bed region using more accurate anatomical targeting"""
    h, w = image.shape[:2]
    
    # Use center-weighted ROI (nail bed is usually centered in proper photos)
    center_x, center_y = w // 2, h // 2
    
    # Extract central portion of nail (avoiding edges) - medical assessment focuses on central nail bed
    roi_width = int(w * 0.6)
    roi_height = int(h * 0.4)
    
    x1 = center_x - roi_width // 2
    y1 = center_y - int(roi_height * 0.3)  # Shifted slightly up from center
    x2 = center_x + roi_width // 2
    y2 = y1 + roi_height
    
    # Apply bounds checking
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    return image[y1:y2, x1:x2]

def extract_conjunctiva_roi(image):
    """Extract palpebral conjunctiva region - specifically targeting lower eyelid"""
    h, w = image.shape[:2]
    
    # Conjunctiva assessment focuses on lower eyelid in medical exams
    roi_width = int(w * 0.7)
    roi_height = int(h * 0.3)
    
    # Target lower portion of eye image where conjunctiva is located
    center_x = w // 2
    y_offset = int(h * 0.6)  # Focus on lower part of image
    
    x1 = center_x - roi_width // 2
    y1 = y_offset
    x2 = center_x + roi_width // 2
    y2 = y1 + roi_height
    
    # Apply bounds checking
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    return image[y1:y2, x1:x2]

def calculate_pallor_score(roi, region_type):
    """Calculate pallor score using clinically relevant color metrics
    
    In clinical assessment:
    - For nails: pink/red indicates healthy, white/pale indicates anemia
    - For conjunctiva: red/pink indicates healthy, pale/white indicates anemia
    
    Lower scores indicate more pallor (more anemic appearance)
    """
    # Convert to different color spaces for comprehensive analysis
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    
    # Extract channels
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    
    # Calculate metrics based on region type
    if region_type == "nail":
        # Extract BGR channels
        bgr_b, bgr_g, bgr_r = cv2.split(roi)
        
        # Calculate pallor metrics
        # 1. Redness in LAB space (a channel: negative=green, positive=red)
        redness = np.mean(a)
        # Remap to 0-100 scale (clinical studies show a* values of ~15-20 for healthy nails)
        redness_score = max(0, min(100, (redness + 10) * 3))
        
        # 2. Red-to-white ratio (higher values = more red, less pale)
        red_white_ratio = np.mean(bgr_r) / (np.mean(l) + 1e-5)
        red_white_score = min(100, red_white_ratio * 100)
        
        # 3. Saturation (less saturation = more pale/white)
        saturation_score = min(100, np.mean(s) * 0.4)
        
        # 4. Red-to-blue ratio (higher in healthy nails)
        red_blue_ratio = np.mean(bgr_r) / (np.mean(bgr_b) + 1e-5)
        red_blue_score = min(100, red_blue_ratio * 30)
        
        # Combined weighted score (scaled to 0-100 range)
        # Parameters tuned to be more sensitive to pallor
        score = (0.35 * redness_score + 0.3 * saturation_score + 
                0.2 * red_white_score + 0.15 * red_blue_score)
                
        # Apply non-linear adjustment to better detect paleness
        # This makes the algorithm more sensitive to changes in the lower range
        if score < 70:
            score = score * 0.8  # Reduce scores for pale samples
        
    else:  # conjunctiva
        # Extract BGR channels
        bgr_b, bgr_g, bgr_r = cv2.split(roi)
        
        # Calculate pallor metrics for conjunctiva (more sensitive than nail bed)
        # 1. Redness in LAB space
        redness = np.mean(a)
        redness_score = max(0, min(100, (redness + 10) * 3))
        
        # 2. Red-to-total ratio
        red_ratio = np.mean(bgr_r) / (np.mean(bgr_r) + np.mean(bgr_g) + np.mean(bgr_b) + 1e-5)
        red_ratio_score = min(100, red_ratio * 200)
        
        # 3. Saturation (key indicator for pallor)
        saturation_score = min(100, np.mean(s) * 0.5)
        
        # 4. Lightness (higher L = more pale/white)
        # Invert so lower lightness (less pale) = higher score
        lightness = np.mean(l)
        lightness_score = max(0, 100 - (lightness - 100) * 0.7)
        
        # Combined weighted score (scaled to 0-100 range)
        # Conjunctiva is more sensitive to pallor than nails
        score = (0.3 * redness_score + 0.3 * red_ratio_score + 
                0.2 * saturation_score + 0.2 * lightness_score)
        
        # Apply non-linear adjustment for better clinical correlation
        if score < 65:
            score = score * 0.75  # Make pale conjunctiva even more indicative of anemia
    
    # Normalize to 0-100 scale with clinically relevant bounds
    normalized_score = max(0, min(100, score))
    
    # Debug printing to identify issues with pale images
    print(f"DEBUG - {region_type} analysis:")
    if region_type == "nail":
        print(f"  Redness (a*): {redness:.2f}, Score: {redness_score:.2f}")
        print(f"  Saturation: {np.mean(s):.2f}, Score: {saturation_score:.2f}")
        print(f"  Red/White ratio: {red_white_ratio:.2f}, Score: {red_white_score:.2f}")
        print(f"  Red/Blue ratio: {red_blue_ratio:.2f}, Score: {red_blue_score:.2f}")
    else:
        print(f"  Redness (a*): {redness:.2f}, Score: {redness_score:.2f}")
        print(f"  Red ratio: {red_ratio:.2f}, Score: {red_ratio_score:.2f}")
        print(f"  Saturation: {np.mean(s):.2f}, Score: {saturation_score:.2f}")
        print(f"  Lightness: {lightness:.2f}, Score: {lightness_score:.2f}")
    print(f"  Final score: {normalized_score:.2f}/100")
    
    return normalized_score

def determine_hemoglobin_estimate(nail_score, conjunctiva_score):
    """Estimate hemoglobin level based on pallor scores
    
    Note: This is an approximation and should be confirmed with blood tests
    Normal Hgb: 
    - Adult males: 13.5-17.5 g/dL
    - Adult females: 12.0-15.5 g/dL
    - Children: varies by age
    
    Lower thresholds provide more sensitivity for detecting anemia
    """
    # Combined weighted score (conjunctiva is more reliable for clinical assessment)
    combined_score = 0.4 * nail_score + 0.6 * conjunctiva_score
    
    # Approximate mapping of score to hemoglobin (g/dL)
    # Adjusted thresholds to be more sensitive to pallor
    if combined_score >= 80:
        return "≥ 14 g/dL (likely normal)"
    elif combined_score >= 65:
        return "12-14 g/dL (possible mild anemia)"
    elif combined_score >= 45:
        return "9-12 g/dL (moderate anemia likely)"
    elif combined_score >= 25:
        return "6-9 g/dL (severe anemia likely)"
    else:
        return "< 6 g/dL (critical anemia possible)"

def determine_diagnosis(nail_score, conjunctiva_score):
    """Determine diagnosis based on clinically validated thresholds
    
    More sensitive thresholds for better detection of mild and moderate anemia
    """
    # Weight conjunctiva more heavily as it's clinically more reliable
    weighted_score = 0.4 * nail_score + 0.6 * conjunctiva_score
    
    # Clinical thresholds with increased sensitivity for pallor detection
    if weighted_score >= 80:
        return "Normal conjunctiva and nail bed coloration - anemia unlikely"
    elif weighted_score >= 65:
        return "Mild pallor detected - possible mild anemia"
    elif weighted_score >= 45:
        return "Moderate pallor detected - moderate anemia probable"
    elif weighted_score >= 25:
        return "Severe pallor detected - severe anemia likely"
    else:
        return "Critical pallor detected - severe anemia highly likely, urgent evaluation needed"

def get_recommendations(weighted_score):
    """Generate clinically appropriate recommendations with adjusted thresholds"""
    if weighted_score >= 80:
        return "No specific action needed based on visual assessment."
    elif weighted_score >= 65:
        return "Consider routine blood tests including CBC, serum ferritin, and iron studies at next clinical visit."
    elif weighted_score >= 45:
        return "Medical evaluation recommended. Complete blood count (CBC), iron studies, and clinical assessment advised."
    elif weighted_score >= 25:
        return "Prompt medical evaluation needed. Complete blood count, iron studies, and clinical assessment advised. Consider evaluation for acute or chronic blood loss."
    else:
        return "Urgent medical attention recommended. This degree of pallor suggests significant anemia that may require immediate intervention."

def generate_medical_report(nail_score, conjunctiva_score, nail_quality, conjunctiva_quality):
    """Generate detailed medical report with clinical context"""
    weighted_score = 0.4 * nail_score + 0.6 * conjunctiva_score
    
    return {
        'examination_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'nail_bed_score': float(nail_score),
        'conjunctiva_score': float(conjunctiva_score),
        'weighted_clinical_score': float(weighted_score),
        'estimated_hemoglobin': determine_hemoglobin_estimate(nail_score, conjunctiva_score),
        'diagnosis': determine_diagnosis(nail_score, conjunctiva_score),
        'recommendation': get_recommendations(weighted_score),
        'limitations': "This is an AI-assisted screening tool only. Clinical correlation and laboratory testing are required for diagnosis.",
        'image_quality': {
            'nail_bed_quality': nail_quality,
            'conjunctiva_quality': conjunctiva_quality
        }
    }

def visualize_analysis(original_nail, original_conjunctiva, nail_roi, conjunctiva_roi, 
                       nail_score, conjunctiva_score, save_path="anemia_analysis_visual.jpg"):
    """Create visual representation of analysis for clinical review"""
    # Create ROI visualization on original images
    nail_viz = original_nail.copy()
    conjunctiva_viz = original_conjunctiva.copy()
    
    # Get ROI coordinates (simplified for demonstration)
    h_nail, w_nail = original_nail.shape[:2]
    h_conj, w_conj = original_conjunctiva.shape[:2]
    
    # Draw rectangle around approximate ROI areas
    nail_roi_y, nail_roi_x = int(h_nail*0.3), int(w_nail*0.2)
    cv2.rectangle(nail_viz, 
                 (nail_roi_x, nail_roi_y), 
                 (w_nail-nail_roi_x, h_nail-nail_roi_y), 
                 (0, 255, 0), 2)
    
    conj_roi_y, conj_roi_x = int(h_conj*0.5), int(w_conj*0.15)
    cv2.rectangle(conjunctiva_viz, 
                 (conj_roi_x, conj_roi_y), 
                 (w_conj-conj_roi_x, h_conj-conj_roi_y), 
                 (0, 255, 0), 2)
    
    # Add score text
    cv2.putText(nail_viz, f"Score: {nail_score:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(conjunctiva_viz, f"Score: {conjunctiva_score:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Resize for consistent display
    nail_viz = cv2.resize(nail_viz, (500, int(500*h_nail/w_nail)))
    conjunctiva_viz = cv2.resize(conjunctiva_viz, (500, int(500*h_conj/w_conj)))
    nail_roi_viz = cv2.resize(nail_roi, (250, int(250*nail_roi.shape[0]/nail_roi.shape[1])))
    conj_roi_viz = cv2.resize(conjunctiva_roi, (250, int(250*conjunctiva_roi.shape[0]/conjunctiva_roi.shape[1])))
    
    # Create result visualization
    h1, w1 = nail_viz.shape[:2]
    h2, w2 = conjunctiva_viz.shape[:2]
    h_roi1, w_roi1 = nail_roi_viz.shape[:2]
    h_roi2, w_roi2 = conj_roi_viz.shape[:2]
    
    # Create blank canvas
    canvas_h = max(h1 + h_roi1, h2 + h_roi2) + 80
    canvas_w = w1 + w2 + 40
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
   
    cv2.putText(canvas, "Anemia Screening Analysis", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add images
    y_offset = 60
    canvas[y_offset:y_offset+h1, 20:20+w1] = nail_viz
    canvas[y_offset:y_offset+h2, 20+w1+20:20+w1+20+w2] = conjunctiva_viz
    
    # Add ROIs
    canvas[y_offset+h1+20:y_offset+h1+20+h_roi1, 20:20+w_roi1] = nail_roi_viz
    canvas[y_offset+h2+20:y_offset+h2+20+h_roi2, 20+w1+20:20+w1+20+w_roi2] = conj_roi_viz
    
    # Add labels
    cv2.putText(canvas, "Nail Bed Analysis", (20, y_offset-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "Conjunctiva Analysis", (20+w1+20, y_offset-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText(canvas, "ROI Analysis", (20, y_offset+h1+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, "ROI Analysis", (20+w1+20, y_offset+h2+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Add weighted score
    weighted_score = 0.4 * nail_score + 0.6 * conjunctiva_score
    cv2.putText(canvas, f"Weighted Clinical Score: {weighted_score:.1f}/100", 
                (20, canvas_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Save visualization
    cv2.imwrite(save_path, canvas)
    return save_path

def main():
    try:
        print("=== Clinical Anemia Screening (Nail Bed & Conjunctiva Analysis) ===")
        print("Loading and analyzing images...")
        
        # Load the specific images
        nail_bed_img, conjunctiva_img = load_specific_images()
        
        # Print basic image info for debugging
        print(f"Nail bed image shape: {nail_bed_img.shape}, dtype: {nail_bed_img.dtype}")
        print(f"Conjunctiva image shape: {conjunctiva_img.shape}, dtype: {conjunctiva_img.dtype}")
        
        # Check for very bright/white images that might indicate pallor
        nail_avg_color = np.mean(nail_bed_img, axis=(0,1))
        conj_avg_color = np.mean(conjunctiva_img, axis=(0,1))
        print(f"Average nail bed color (BGR): {nail_avg_color}")
        print(f"Average conjunctiva color (BGR): {conj_avg_color}")
        
        # Apply pre-check for extremely pale images
        nail_brightness = np.mean(nail_avg_color)
        conj_brightness = np.mean(conj_avg_color)
        
        # Early detection of very pale images
        is_nail_very_pale = nail_brightness > 200
        is_conj_very_pale = conj_brightness > 200
        
        if is_nail_very_pale:
            print("NOTICE: Nail bed appears very pale/white in initial analysis")
        if is_conj_very_pale:
            print("NOTICE: Conjunctiva appears very pale/white in initial analysis")
        
        # Analyze image quality
        nail_quality = analyze_image_quality(nail_bed_img)
        conjunctiva_quality = analyze_image_quality(conjunctiva_img)
        
        # Quality warnings
        if nail_quality['sharpness'] < 50 or nail_quality['contrast'] < 40:
            print("WARNING: Nail bed image quality may be insufficient for accurate assessment")
        if conjunctiva_quality['sharpness'] < 50 or conjunctiva_quality['contrast'] < 40:
            print("WARNING: Conjunctiva image quality may be insufficient for accurate assessment")
        
        # Enhance images with medical-grade processing
        enhanced_nail = enhance_image_for_medical_analysis(nail_bed_img)
        enhanced_conjunctiva = enhance_image_for_medical_analysis(conjunctiva_img)
        
        # Extract anatomically relevant ROIs
        nail_roi = extract_nail_bed_roi(enhanced_nail)
        conjunctiva_roi = extract_conjunctiva_roi(enhanced_conjunctiva)
        
        print("\n--- Performing pallor analysis ---")
        
        # Calculate clinical scores
        nail_score = calculate_pallor_score(nail_roi, "nail")
        conjunctiva_score = calculate_pallor_score(conjunctiva_roi, "conjunctiva")
        
        # Override scores if images are extremely pale (backup mechanism)
        if is_nail_very_pale and nail_score > 60:
            nail_score = min(nail_score, 60)  # Cap score for very pale nails
            print("Applied adjustment for very pale nail bed")
            
        if is_conj_very_pale and conjunctiva_score > 55:
            conjunctiva_score = min(conjunctiva_score, 55)  # Cap score for very pale conjunctiva
            print("Applied adjustment for very pale conjunctiva")
        
        # Generate report
        report = generate_medical_report(nail_score, conjunctiva_score, nail_quality, conjunctiva_quality)
        
        # Generate visualization
        viz_path = visualize_analysis(nail_bed_img, conjunctiva_img, nail_roi, conjunctiva_roi, 
                                     nail_score, conjunctiva_score)
        
        # Save results
        with open("anemia_screening_report.json", "w") as f:
            json.dump(report, f, indent=4)
        
        print("\n=== Clinical Results ===")
        print(f"Nail Bed Pallor Assessment: {nail_score:.1f}/100")
        print(f"Conjunctiva Pallor Assessment: {conjunctiva_score:.1f}/100")
        print(f"Weighted Clinical Score: {report['weighted_clinical_score']:.1f}/100")
        print(f"\nEstimated Hemoglobin: {report['estimated_hemoglobin']}")
        print(f"\nClinical Impression: {report['diagnosis']}")
        print(f"Recommendation: {report['recommendation']}")
        print(f"\nAnalysis visualization saved to '{viz_path}'")
        print("Full report saved to 'anemia_screening_report.json'")
        print("\nIMPORTANT: This is a screening tool only. Clinical correlation and laboratory testing are required for diagnosis.")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("\nPossible solutions:")
        print("- Ensure images are available and properly formatted")
        print("- Verify images contain clear close-ups of nail bed and palpebral conjunctiva")
        print("- Ensure good lighting and focus in images")
        print("- Use unfiltered images without digital enhancement")

if __name__ == "__main__":
    main()