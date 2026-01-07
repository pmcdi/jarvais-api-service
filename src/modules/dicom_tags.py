"""
DICOM Tags Module.

This module defines a curated list of clinically important DICOM tags
for medical imaging data analysis. The tags cover patient demographics,
study information, imaging parameters, and acquisition metadata.

Reference: DICOM Standard PS3.6 - Data Dictionary
https://dicom.nema.org/medical/dicom/current/output/html/part06.html
"""

from typing import Dict, List, NamedTuple


class DicomTagInfo(NamedTuple):
    """Information about a DICOM tag."""
    tag: str  # DICOM tag in format (XXXX,XXXX)
    keyword: str  # DICOM keyword
    description: str  # Human-readable description
    category: str  # Categorization for grouping


# Curated list of clinically important DICOM tags
DICOM_TAGS: List[DicomTagInfo] = [
    # ==========================================================================
    # PATIENT DEMOGRAPHICS
    # ==========================================================================
    DicomTagInfo("(0010,0010)", "PatientName", "Patient's full name", "demographics"),
    DicomTagInfo("(0010,0020)", "PatientID", "Primary hospital identification number", "demographics"),
    DicomTagInfo("(0010,0030)", "PatientBirthDate", "Patient's date of birth", "demographics"),
    DicomTagInfo("(0010,0040)", "PatientSex", "Patient's sex (M/F/O)", "demographics"),
    DicomTagInfo("(0010,1010)", "PatientAge", "Patient's age at acquisition", "demographics"),
    DicomTagInfo("(0010,1020)", "PatientSize", "Patient's height in meters", "demographics"),
    DicomTagInfo("(0010,1030)", "PatientWeight", "Patient's weight in kilograms", "demographics"),
    DicomTagInfo("(0010,21B0)", "AdditionalPatientHistory", "Additional patient history", "demographics"),
    DicomTagInfo("(0010,2160)", "EthnicGroup", "Patient's ethnic group", "demographics"),
    
    # ==========================================================================
    # STUDY INFORMATION
    # ==========================================================================
    DicomTagInfo("(0008,0020)", "StudyDate", "Date the study was performed", "study"),
    DicomTagInfo("(0008,0030)", "StudyTime", "Time the study was performed", "study"),
    DicomTagInfo("(0008,0050)", "AccessionNumber", "RIS/HIS accession number", "study"),
    DicomTagInfo("(0008,0060)", "Modality", "Type of equipment (CT/MR/US/etc)", "study"),
    DicomTagInfo("(0008,0070)", "Manufacturer", "Equipment manufacturer", "study"),
    DicomTagInfo("(0008,0080)", "InstitutionName", "Institution where study performed", "study"),
    DicomTagInfo("(0008,0090)", "ReferringPhysicianName", "Referring physician's name", "study"),
    DicomTagInfo("(0008,1030)", "StudyDescription", "Description of the study", "study"),
    DicomTagInfo("(0008,103E)", "SeriesDescription", "Description of the series", "study"),
    DicomTagInfo("(0008,1050)", "PerformingPhysicianName", "Physician performing the study", "study"),
    DicomTagInfo("(0020,000D)", "StudyInstanceUID", "Unique identifier for the study", "study"),
    DicomTagInfo("(0020,000E)", "SeriesInstanceUID", "Unique identifier for the series", "study"),
    DicomTagInfo("(0020,0010)", "StudyID", "User-defined study ID", "study"),
    DicomTagInfo("(0020,0011)", "SeriesNumber", "Number identifying the series", "study"),
    DicomTagInfo("(0020,0013)", "InstanceNumber", "Number identifying the image", "study"),
    
    # ==========================================================================
    # IMAGING PARAMETERS - GENERAL
    # ==========================================================================
    DicomTagInfo("(0018,0015)", "BodyPartExamined", "Body part being examined", "imaging"),
    DicomTagInfo("(0018,0050)", "SliceThickness", "Nominal slice thickness in mm", "imaging"),
    DicomTagInfo("(0018,0088)", "SpacingBetweenSlices", "Spacing between slices in mm", "imaging"),
    DicomTagInfo("(0018,1030)", "ProtocolName", "Name of acquisition protocol", "imaging"),
    DicomTagInfo("(0018,5100)", "PatientPosition", "Patient position (HFS/HFP/etc)", "imaging"),
    DicomTagInfo("(0028,0010)", "Rows", "Number of rows in the image", "imaging"),
    DicomTagInfo("(0028,0011)", "Columns", "Number of columns in the image", "imaging"),
    DicomTagInfo("(0028,0030)", "PixelSpacing", "Physical distance between pixels", "imaging"),
    DicomTagInfo("(0028,0100)", "BitsAllocated", "Bits allocated per pixel", "imaging"),
    DicomTagInfo("(0028,0101)", "BitsStored", "Bits stored per pixel", "imaging"),
    DicomTagInfo("(0028,1050)", "WindowCenter", "Window center for display", "imaging"),
    DicomTagInfo("(0028,1051)", "WindowWidth", "Window width for display", "imaging"),
    
    # ==========================================================================
    # CT-SPECIFIC PARAMETERS
    # ==========================================================================
    DicomTagInfo("(0018,0060)", "KVP", "Peak kilovoltage output", "ct"),
    DicomTagInfo("(0018,1150)", "ExposureTime", "Time of X-ray exposure in ms", "ct"),
    DicomTagInfo("(0018,1151)", "XRayTubeCurrent", "X-ray tube current in mA", "ct"),
    DicomTagInfo("(0018,1152)", "Exposure", "Exposure in mAs", "ct"),
    DicomTagInfo("(0018,0022)", "ScanOptions", "CT scan options", "ct"),
    DicomTagInfo("(0018,0090)", "DataCollectionDiameter", "Data collection diameter", "ct"),
    DicomTagInfo("(0018,1100)", "ReconstructionDiameter", "Reconstruction diameter", "ct"),
    DicomTagInfo("(0018,1210)", "ConvolutionKernel", "Reconstruction kernel", "ct"),
    DicomTagInfo("(0018,9345)", "CTDIvol", "CT Dose Index Volume", "ct"),
    DicomTagInfo("(0018,115E)", "ImageAndFluoroscopyAreaDoseProduct", "Dose Area Product", "ct"),
    
    # ==========================================================================
    # MRI-SPECIFIC PARAMETERS
    # ==========================================================================
    DicomTagInfo("(0018,0020)", "ScanningSequence", "Type of MR sequence (SE/GR/etc)", "mri"),
    DicomTagInfo("(0018,0021)", "SequenceVariant", "Sequence variant", "mri"),
    DicomTagInfo("(0018,0023)", "MRAcquisitionType", "2D or 3D acquisition", "mri"),
    DicomTagInfo("(0018,0024)", "SequenceName", "Name of the pulse sequence", "mri"),
    DicomTagInfo("(0018,0080)", "RepetitionTime", "Repetition time (TR) in ms", "mri"),
    DicomTagInfo("(0018,0081)", "EchoTime", "Echo time (TE) in ms", "mri"),
    DicomTagInfo("(0018,0082)", "InversionTime", "Inversion time (TI) in ms", "mri"),
    DicomTagInfo("(0018,0083)", "NumberOfAverages", "Number of signal averages", "mri"),
    DicomTagInfo("(0018,0084)", "ImagingFrequency", "Imaging frequency in MHz", "mri"),
    DicomTagInfo("(0018,0085)", "ImagedNucleus", "Nucleus that is imaged", "mri"),
    DicomTagInfo("(0018,0087)", "MagneticFieldStrength", "Field strength in Tesla", "mri"),
    DicomTagInfo("(0018,0091)", "EchoTrainLength", "Number of echoes in train", "mri"),
    DicomTagInfo("(0018,0093)", "PercentSampling", "Fraction of acquisition matrix", "mri"),
    DicomTagInfo("(0018,0094)", "PercentPhaseFieldOfView", "Phase FOV percentage", "mri"),
    DicomTagInfo("(0018,1310)", "AcquisitionMatrix", "Acquisition matrix dimensions", "mri"),
    DicomTagInfo("(0018,1312)", "InPlanePhaseEncodingDirection", "Phase encoding direction", "mri"),
    DicomTagInfo("(0018,1314)", "FlipAngle", "Flip angle in degrees", "mri"),
    DicomTagInfo("(0018,0095)", "PixelBandwidth", "Pixel bandwidth in Hz/pixel", "mri"),
    DicomTagInfo("(0019,100A)", "NumberOfImagesInMosaic", "Number of images in mosaic (Siemens)", "mri"),
    
    # ==========================================================================
    # ULTRASOUND-SPECIFIC PARAMETERS
    # ==========================================================================
    DicomTagInfo("(0018,6011)", "SequenceOfUltrasoundRegions", "Ultrasound region sequence", "us"),
    DicomTagInfo("(0018,602C)", "PhysicalDeltaX", "Physical distance per pixel X", "us"),
    DicomTagInfo("(0018,602E)", "PhysicalDeltaY", "Physical distance per pixel Y", "us"),
    
    # ==========================================================================
    # RADIOTHERAPY PARAMETERS
    # ==========================================================================
    DicomTagInfo("(300A,00B2)", "TreatmentMachineName", "Treatment machine name", "rt"),
    DicomTagInfo("(300A,00C6)", "RadiationType", "Type of radiation", "rt"),
    DicomTagInfo("(300A,011E)", "GantryAngle", "Gantry angle in degrees", "rt"),
    DicomTagInfo("(300C,0006)", "ReferencedBeamNumber", "Referenced beam number", "rt"),
    
    # ==========================================================================
    # CLINICAL/DIAGNOSTIC INFORMATION
    # ==========================================================================
    DicomTagInfo("(0008,1080)", "AdmittingDiagnosesDescription", "Admitting diagnoses", "clinical"),
    DicomTagInfo("(0010,2000)", "MedicalAlerts", "Medical alerts", "clinical"),
    DicomTagInfo("(0010,2110)", "Allergies", "Patient allergies", "clinical"),
    DicomTagInfo("(0010,21C0)", "PregnancyStatus", "Pregnancy status", "clinical"),
    DicomTagInfo("(0010,4000)", "PatientComments", "Comments about the patient", "clinical"),
    DicomTagInfo("(0032,1030)", "ReasonForStudy", "Reason for the study", "clinical"),
    DicomTagInfo("(0032,1060)", "RequestedProcedureDescription", "Requested procedure", "clinical"),
    DicomTagInfo("(0040,0007)", "ScheduledProcedureStepDescription", "Scheduled procedure step", "clinical"),
    DicomTagInfo("(0040,1001)", "RequestedProcedureID", "Requested procedure ID", "clinical"),
    
    # ==========================================================================
    # CONTRAST AGENT INFORMATION
    # ==========================================================================
    DicomTagInfo("(0018,0010)", "ContrastBolusAgent", "Contrast agent name", "contrast"),
    DicomTagInfo("(0018,0012)", "ContrastBolusAgentSequence", "Contrast agent sequence", "contrast"),
    DicomTagInfo("(0018,1040)", "ContrastBolusRoute", "Route of administration", "contrast"),
    DicomTagInfo("(0018,1041)", "ContrastBolusVolume", "Volume of contrast agent", "contrast"),
    DicomTagInfo("(0018,1042)", "ContrastBolusStartTime", "Time contrast injection started", "contrast"),
    DicomTagInfo("(0018,1043)", "ContrastBolusStopTime", "Time contrast injection stopped", "contrast"),
    DicomTagInfo("(0018,1044)", "ContrastBolusTotalDose", "Total dose of contrast agent", "contrast"),
    DicomTagInfo("(0018,1046)", "ContrastFlowRate", "Rate of contrast injection", "contrast"),
    
    # ==========================================================================
    # IMAGE ACQUISITION TIMING
    # ==========================================================================
    DicomTagInfo("(0008,0021)", "SeriesDate", "Date the series was acquired", "timing"),
    DicomTagInfo("(0008,0022)", "AcquisitionDate", "Date the image was acquired", "timing"),
    DicomTagInfo("(0008,0023)", "ContentDate", "Date image pixel data created", "timing"),
    DicomTagInfo("(0008,0031)", "SeriesTime", "Time the series was acquired", "timing"),
    DicomTagInfo("(0008,0032)", "AcquisitionTime", "Time the image was acquired", "timing"),
    DicomTagInfo("(0008,0033)", "ContentTime", "Time image pixel data created", "timing"),
    DicomTagInfo("(0018,1060)", "TriggerTime", "Time delay from trigger", "timing"),
    DicomTagInfo("(0020,1041)", "SliceLocation", "Relative position of slice", "timing"),
    
    # ==========================================================================
    # EQUIPMENT/SOFTWARE INFORMATION
    # ==========================================================================
    DicomTagInfo("(0008,0080)", "InstitutionName", "Name of institution", "equipment"),
    DicomTagInfo("(0008,0081)", "InstitutionAddress", "Address of institution", "equipment"),
    DicomTagInfo("(0008,1010)", "StationName", "Name of the station", "equipment"),
    DicomTagInfo("(0008,1040)", "InstitutionalDepartmentName", "Department name", "equipment"),
    DicomTagInfo("(0008,1090)", "ManufacturerModelName", "Model name of equipment", "equipment"),
    DicomTagInfo("(0018,1000)", "DeviceSerialNumber", "Equipment serial number", "equipment"),
    DicomTagInfo("(0018,1020)", "SoftwareVersions", "Software versions", "equipment"),
]


# Create lookup dictionaries for easy access
DICOM_TAGS_BY_KEYWORD: Dict[str, DicomTagInfo] = {
    tag.keyword: tag for tag in DICOM_TAGS
}

DICOM_TAGS_BY_TAG: Dict[str, DicomTagInfo] = {
    tag.tag: tag for tag in DICOM_TAGS
}

DICOM_TAGS_BY_CATEGORY: Dict[str, List[DicomTagInfo]] = {}
for tag in DICOM_TAGS:
    if tag.category not in DICOM_TAGS_BY_CATEGORY:
        DICOM_TAGS_BY_CATEGORY[tag.category] = []
    DICOM_TAGS_BY_CATEGORY[tag.category].append(tag)


# List of just the keywords for validation
VALID_DICOM_KEYWORDS: List[str] = list(DICOM_TAGS_BY_KEYWORD.keys())


def get_tag_info(keyword_or_tag: str) -> DicomTagInfo | None:
    """
    Get DICOM tag information by keyword or tag.
    
    Args:
        keyword_or_tag: DICOM keyword (e.g., "PatientID") or tag (e.g., "(0010,0020)")
        
    Returns:
        DicomTagInfo if found, None otherwise
    """
    if keyword_or_tag in DICOM_TAGS_BY_KEYWORD:
        return DICOM_TAGS_BY_KEYWORD[keyword_or_tag]
    if keyword_or_tag in DICOM_TAGS_BY_TAG:
        return DICOM_TAGS_BY_TAG[keyword_or_tag]
    return None


def get_tags_by_category(category: str) -> List[DicomTagInfo]:
    """
    Get all DICOM tags in a specific category.
    
    Args:
        category: Category name (demographics, study, imaging, ct, mri, us, rt, 
                  clinical, contrast, timing, equipment)
        
    Returns:
        List of DicomTagInfo for the specified category
    """
    return DICOM_TAGS_BY_CATEGORY.get(category, [])


def get_all_categories() -> List[str]:
    """
    Get list of all available DICOM tag categories.
    
    Returns:
        List of category names
    """
    return list(DICOM_TAGS_BY_CATEGORY.keys())
