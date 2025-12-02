"""
Enhanced Construction Project Dashboard - FastAPI Backend
==========================================================
Production-ready API with comprehensive validation, security, and all dashboard endpoints.
FIXED: Now correctly reads 'Award' column from Excel as contractor data.
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import FileResponse
from enum import Enum
import pandas as pd
from collections import defaultdict
import os
import logging
from pathlib import Path
import logging
logger = logging.getLogger("app")

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS
# ============================================================================

class StatusEnum(str, Enum):
    ON_GOING = "On-going"
    COMPLETED = "Completed"
    PENDING = "Pending"

class BUEnum(str, Enum):
    FARM = "Farm"
    FOOD = "Food"
    FEED = "Feed"
    SWINE = "Swine"

class UnitEnum(str, Enum):
    METER = "m"
    SQUARE_METER = "m²"
    CUBIC_METER = "m³"
    NUMBER = "No."
    LUMP_SUM = "LS"
    UNIT = "unit"
    LOT = "lot"

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class Project(BaseModel):
    """Complete project model with validation"""
    id: int
    project: str
    bu: str
    location: str
    status: str
    percentage: float = Field(..., alias="percent", ge=0, le=100)
    value_rm: float = Field(..., ge=0)
    value_completed: float = Field(..., ge=0)
    vo: Optional[float] = Field(0.0, ge=0)
    cds: Optional[str] = None
    estimated_days: Optional[int] = Field(None, ge=0)
    finish_date: Optional[str] = None
    unit: Optional[str] = None
    quantity: Optional[float] = Field(None, ge=0)
    average_rm_unit: Optional[float] = Field(None, ge=0)
    contractor: Optional[str] = None  # This will be populated from 'award' column
    safuan: Optional[float] = Field(None, ge=0)
    hamdee: Optional[float] = Field(None, ge=0)
    woradate: Optional[float] = Field(None, ge=0)
    nantawat: Optional[float] = Field(None, ge=0)
    alya: Optional[float] = Field(None, ge=0)
    vignes: Optional[float] = Field(None, ge=0)
    azim: Optional[float] = Field(None, ge=0)
    
    class Config:
        populate_by_name = True

    @field_validator('value_completed')
    @classmethod
    def validate_completed_value(cls, v, values):
        """Ensure completed value doesn't exceed total value"""
        if 'value_rm' in values and v > values['value_rm']:
            logger.warning(f"Completed value {v} exceeds total value {values['value_rm']}")
        return v

class ProjectSummaryKPI(BaseModel):
    """Top-level dashboard KPIs"""
    total_projects: int = Field(..., ge=0)
    total_value_rm: float = Field(..., ge=0)
    total_completed_value: float = Field(..., ge=0)
    average_completion_percentage: float = Field(..., ge=0, le=100)
    total_quantity_installed: float = Field(..., ge=0)
    active_contractors_count: int = Field(..., ge=0)
    completed_projects: int = Field(..., ge=0)
    ongoing_projects: int = Field(..., ge=0)
    pending_projects: int = Field(..., ge=0)
    completion_vs_remaining: Dict[str, float]
    total_variation_orders: float = Field(..., ge=0)

class ProjectProgress(BaseModel):
    """Project progress visualization data"""
    id: int
    project: str
    progress_percent: float = Field(..., ge=0, le=100)
    value_rm: float
    value_completed: float
    contractor: str
    award: str  # Add this field for frontend compatibility
    status: str
    color_code: str
    bu: str
    location: str

class CostAnalysis(BaseModel):
    """Cost analysis with efficiency metrics"""
    id: int
    project: str
    bu: str
    value_rm: float
    value_completed: float
    variation_order: float
    remaining_value: float
    cost_efficiency: float
    contractor: str
    status: str

class ContractorKPI(BaseModel):
    """Contractor performance metrics"""
    contractor: str
    total_value: float
    total_completed_value: float
    average_cost_per_unit: float
    completion_percentage: float
    number_of_projects: int
    total_quantity: float
    ongoing_projects: int
    completed_projects: int
    efficiency_score: float

class UnitAnalysis(BaseModel):
    """Unit-wise work analysis"""
    unit: str
    total_quantity: float
    total_value: float
    average_cost_per_unit: float
    project_count: int
    min_cost_per_unit: float
    max_cost_per_unit: float

class CDSTracking(BaseModel):
    """CDS tracking with timeline"""
    id: int
    cds: str
    project: str
    bu: str
    estimated_days: int
    finish_date: Optional[str]
    status: str
    completion_percent: float
    contractor: str
    days_elapsed: Optional[int] = None

class BUAnalysis(BaseModel):
    """Business Unit comprehensive analysis"""
    bu: str
    total_value: float
    total_completed_value: float
    total_vo: float
    project_count: int
    average_completion: float
    completed_projects: int
    ongoing_projects: int
    pending_projects: int
    efficiency: float

class LocationAnalysis(BaseModel):
    """Location-wise project distribution"""
    location: str
    bu: str
    project_count: int
    total_value: float
    total_completed_value: float
    contractors: List[str]
    average_completion: float
    status_distribution: Dict[str, int]

class StaffWorkload(BaseModel):
    """Staff workload and performance"""
    staff_name: str
    total_value_supervised: float
    projects_under_supervision: int
    average_completion: float
    total_completed_value: float
    efficiency: float
    project_list: List[str]

class TimelineData(BaseModel):
    """Gantt chart timeline data"""
    id: int
    project: str
    contractor: str
    bu: str
    planned_finish_date: Optional[str]
    estimated_days: Optional[int]
    status: str
    completion_percent: float
    start_date: Optional[str] = None
    delay_days: Optional[int] = None

class FilterOptions(BaseModel):
    """All available filter options"""
    projects: List[str]
    contractors: List[str]
    bus: List[str]
    statuses: List[str]
    locations: List[str]
    units: List[str]
    cds_list: List[str]
    months: List[str]
    years: List[int]

class BubbleChartData(BaseModel):
    """Bubble chart visualization data"""
    unit: str
    bubble_size: float
    average_cost_per_unit: float
    project_count: int
    total_value: float
    color_category: str

class ProjectDetail(BaseModel):
    """Complete project details for drill-down"""
    project: Project
    related_contractors: List[str]
    cost_breakdown: Dict[str, float]
    timeline_info: Dict[str, Any]
    staff_assignments: Dict[str, float]
    performance_metrics: Dict[str, float]

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Construction Project Dashboard API",
    description="Production-ready REST API for comprehensive construction project management",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Dashboard - Summary KPIs", "description": "Top-level dashboard KPIs"},
        {"name": "Dashboard - Project Progress", "description": "Project progress tracking"},
        {"name": "Dashboard - Cost Analysis", "description": "Financial analysis and cost tracking"},
        {"name": "Dashboard - Quantity Analysis", "description": "Quantity and unit analysis"},
        {"name": "Dashboard - CDS Tracking", "description": "CDS and timeline tracking"},
        {"name": "Dashboard - Contractor Performance", "description": "Contractor KPIs and performance"},
        {"name": "Dashboard - Business Unit Analysis", "description": "BU-wise analysis"},
        {"name": "Dashboard - Location Analysis", "description": "Location-wise distribution"},
        {"name": "Dashboard - Staff Workload", "description": "Staff workload and assignments"},
        {"name": "Dashboard - Filters", "description": "Filter options and slicers"},
        {"name": "Projects - CRUD Operations", "description": "Project CRUD endpoints"},
        {"name": "Data Management", "description": "Data loading and management"},
        {"name": "Data Export", "description": "Export functionality"},
        {"name": "Root", "description": "Health check and info"},
    ]
)

# ============================================================================
# CORS CONFIGURATION (UPDATE FOR PRODUCTION)
# ============================================================================

# For development
ALLOWED_ORIGINS = ["*"]

# For production, replace with your actual frontend domain:
# ALLOWED_ORIGINS = ["https://yourdashboard.com", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# STATIC FILES (FRONTEND)
# ============================================================================

# Mount static files directory (for CSS, JS, images, etc. if needed)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the frontend HTML file at root


@app.get("/", response_class=HTMLResponse, tags=["Root"], include_in_schema=False)
async def serve_frontend():
    """Serve the frontend dashboard"""
    try:
        # Try multiple possible paths for index.html
        possible_paths = [
            "index.html",
            os.path.join(os.path.dirname(__file__), "index.html"),
            os.path.join(Path(__file__).parent, "index.html"),
            os.path.join(os.getcwd(), "index.html"),
        ]
        
        for html_path in possible_paths:
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    return f.read()
        
        # If not found, return error
        raise FileNotFoundError("index.html not found in any expected location")
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <head><title>Dashboard Not Found</title></head>
                <body>
                    <h1>Frontend Not Found</h1>
                    <p>index.html file not found. Please ensure it exists in the project directory.</p>
                    <p><a href="/docs">View API Documentation</a></p>
                </body>
            </html>
            """,
            status_code=404
        )

# ============================================================================
# DATA LOADING FROM EXCEL - FIXED TO USE 'AWARD' COLUMN
# ============================================================================


@app.get("/upload.html")
async def serve_upload():
    """Serve upload.html page"""
    try:
        # Try multiple possible paths for upload.html
        possible_paths = [
            "upload.html",
            os.path.join(os.path.dirname(__file__), "..", "upload.html"),
            os.path.join(Path(__file__).parent.parent, "upload.html"),
            os.path.join(os.getcwd(), "upload.html"),
        ]
        
        for html_path in possible_paths:
            if os.path.exists(html_path):
                return FileResponse(html_path)
        
        # If not found, return error
        raise FileNotFoundError("upload.html not found in any expected location")
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <head><title>Upload Page Not Found</title></head>
                <body>
                    <h1>Upload Page Not Found</h1>
                    <p>upload.html file not found. Please ensure it exists in the project directory.</p>
                    <p><a href="/">Back to Dashboard</a> | <a href="/docs">API Documentation</a></p>
                </body>
            </html>
            """,
            status_code=404
        )


# app.py is in DASHBOARD/api/app.py → parent.parent == repo/DASHBOARD
BASE_DIR = Path(__file__).parent.parent
TEMPLATE_PATH = BASE_DIR / "template.xlsx"

@app.get("/api/v1/download-template")
def download_template():
    if not TEMPLATE_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Template not found at {TEMPLATE_PATH}"
        )
    return FileResponse(
        TEMPLATE_PATH,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="BuildPro_Construction_Template.xlsx",
    )



def load_excel_data(file_path: str) -> List[Dict]:
    """
    Load and validate project data from Excel file.
    FIXED: Now correctly maps 'Award' column to 'contractor' field.
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        List of validated project dictionaries
        
    Raises:
        ValueError: If data validation fails
    """
    try:
        df = pd.read_excel(file_path)
        
        logger.info(f"📊 Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        
        logger.info(f"📋 Columns found: {list(df.columns)}")
        
        # FIXED: Column mapping - Map 'award' to 'contractor'
        column_mapping = {
            'project': 'project',
            'bu': 'bu',
            'location': 'location',
            'status': 'status',
            '%': 'percent',
            'value_rm': 'value_rm',
            'value_completed': 'value_completed',
            'vorm': 'vo',
            'cds': 'cds',
            'estimated_days': 'estimated_days',
            'finish_date': 'finish_date',
            'unit': 'unit',
            'quantity': 'quantity',
            'average_rm/unit': 'average_rm_unit',
            'award': 'contractor',  # ✅ FIX: Map 'award' column to 'contractor'
            'award_contractor': 'contractor',  # Also support this variant if exists
            'contractor': 'contractor',  # If already named 'contractor', keep it
            'safuan': 'safuan',
            'hamdee': 'hamdee',
            'woradate': 'woradate',
            'nantawat': 'nantawat',
            'alya': 'alya',
            'vignes': 'vignes',
            'azim': 'azim'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        logger.info(f"✅ Column mapping applied. Contractor column exists: {'contractor' in df.columns}")
        
        # Add ID column
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)
        
        # Numeric columns
        numeric_columns = ['percent', 'value_rm', 'value_completed', 'vo', 
                          'estimated_days', 'quantity', 'average_rm_unit',
                          'safuan', 'hamdee', 'woradate', 'nantawat', 
                          'alya', 'vignes', 'azim']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # String columns
        string_columns = ['project', 'bu', 'location', 'status', 'cds', 
                         'finish_date', 'unit', 'contractor']
        
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', None)
        
        # Log contractor data
        if 'contractor' in df.columns:
            unique_contractors = df['contractor'].dropna().unique()
            logger.info(f"🏗️  Found {len(unique_contractors)} unique contractors: {list(unique_contractors)[:5]}...")
        else:
            logger.warning("⚠️  No contractor column found after mapping!")
        
        # Data validation
        required_columns = ['project', 'bu', 'location', 'status', 'percent', 'value_rm']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert to list of dicts
        projects_list = df.to_dict('records')
        
        # Clean up
        for project in projects_list:
            for key, value in project.items():
                if pd.isna(value) or value == 'nan':
                    project[key] = None if key in string_columns else 0
                elif isinstance(value, float) and key not in numeric_columns:
                    project[key] = str(value) if not pd.isna(value) else None
        
        # Remove invalid rows
        projects_list = [p for p in projects_list if p.get('project') and p.get('project') != 'None']
        
        # Count projects with contractors
        projects_with_contractors = len([p for p in projects_list if p.get('contractor') and p.get('contractor') != 'None'])
        
        logger.info(f"✅ Successfully loaded {len(projects_list)} valid projects from Excel")
        logger.info(f"✅ {projects_with_contractors} projects have contractor/award data")
        
        return projects_list
        
    except Exception as e:
        logger.error(f"❌ Error loading Excel file: {e}")
        raise ValueError(f"Failed to load Excel file: {str(e)}")

# ============================================================================
# DATABASE/DATA LAYER
# ============================================================================

EXCEL_FILE_PATH = "projects_data.xlsx"
PROJECTS_DATA = []

def initialize_data():
    """Initialize data from Excel or use sample data"""
    global PROJECTS_DATA
    
    if os.path.exists(EXCEL_FILE_PATH):
        logger.info(f"📂 Found Excel file: {EXCEL_FILE_PATH}")
        try:
            PROJECTS_DATA = load_excel_data(EXCEL_FILE_PATH)
        except Exception as e:
            logger.error(f"Failed to load Excel: {e}")
            PROJECTS_DATA = get_sample_data()
    else:
        logger.warning(f"⚠️  Excel file not found: {EXCEL_FILE_PATH}")
        logger.info("📋 Using sample data...")
        PROJECTS_DATA = get_sample_data()
    
    logger.info(f"📊 Total projects loaded: {len(PROJECTS_DATA)}")

def get_sample_data() -> List[Dict]:
    """Return sample data for testing"""
    return [
        {
            "id": 1,
            "project": "Water Supply Installation - 2km Pipe",
            "bu": "Farm",
            "location": "Jo-Vin Aaehing",
            "status": "Completed",
            "percent": 100.0,
            "value_rm": 134000,
            "value_completed": 134000,
            "vo": 0,
            "cds": "CDS-001",
            "estimated_days": 90,
            "finish_date": "2025-03-04",
            "unit": "m",
            "quantity": 2000.00,
            "average_rm_unit": 67.00,
            "contractor": "Kona Plumbing",
            "safuan": 50000.00,
            "hamdee": None,
            "woradate": 40000.00,
            "nantawat": None,
            "alya": None,
            "vignes": 44000.00,
            "azim": None
        },
        {
            "id": 2,
            "project": "CCTV Installation - 48 units",
            "bu": "Food",
            "location": "Penang Main Factory",
            "status": "On-going",
            "percent": 75.0,
            "value_rm": 95000,
            "value_completed": 71250,
            "vo": 5000,
            "cds": "CDS-002",
            "estimated_days": 60,
            "finish_date": "2025-04-15",
            "unit": "No.",
            "quantity": 48.00,
            "average_rm_unit": 1979.17,
            "contractor": "Zheng Cheng",
            "safuan": None,
            "hamdee": 35000.00,
            "woradate": None,
            "nantawat": 36250.00,
            "alya": None,
            "vignes": None,
            "azim": None
        },
        {
            "id": 3,
            "project": "Fencing Installation - Perimeter",
            "bu": "Farm",
            "location": "Kota Hatchery",
            "status": "Completed",
            "percent": 100.0,
            "value_rm": 316000,
            "value_completed": 316000,
            "vo": 15000,
            "cds": "CDS-003",
            "estimated_days": 45,
            "finish_date": "2025-02-28",
            "unit": "m",
            "quantity": 1000.00,
            "average_rm_unit": 316.00,
            "contractor": "Pioneer",
            "safuan": 100000.00,
            "hamdee": None,
            "woradate": None,
            "nantawat": None,
            "alya": 50000.00,
            "vignes": None,
            "azim": 166000.00
        }
    ]

# Initialize data on startup
initialize_data()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def filter_projects(
    data: List[Dict],
    bu: Optional[str] = None,
    status: Optional[str] = None,
    contractor: Optional[str] = None,
    location: Optional[str] = None,
    unit: Optional[str] = None,
    cds: Optional[str] = None,
    min_progress: Optional[float] = None,
    max_progress: Optional[float] = None
) -> List[Dict]:
    """Apply filters to project data"""
    filtered = data.copy()
    
    if bu:
        filtered = [p for p in filtered if p.get('bu') == bu]
    if status:
        filtered = [p for p in filtered if p.get('status') == status]
    if contractor:
        filtered = [p for p in filtered if p.get('contractor') == contractor]
    if location:
        filtered = [p for p in filtered if p.get('location') == location]
    if unit:
        filtered = [p for p in filtered if p.get('unit') == unit]
    if cds:
        filtered = [p for p in filtered if p.get('cds') == cds]
    if min_progress is not None:
        filtered = [p for p in filtered if (p.get('percent', 0) or 0) >= min_progress]
    if max_progress is not None:
        filtered = [p for p in filtered if (p.get('percent', 0) or 0) <= max_progress]
    
    return filtered

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    return round(numerator / denominator, 2) if denominator > 0 else default

# ============================================================================
# DATA MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/api/v1/data/reload", tags=["Data Management"])
async def reload_data_from_excel(file_path: Optional[str] = None):
    """Reload data from Excel file without restarting server"""
    global PROJECTS_DATA
    
    path = file_path or EXCEL_FILE_PATH
    
    if not os.path.exists(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Excel file not found: {path}"
        )
    
    try:
        new_data = load_excel_data(path)
        PROJECTS_DATA = new_data
        
        # Count contractors
        contractors_count = len(set(p.get('contractor') for p in PROJECTS_DATA if p.get('contractor') and p.get('contractor') != 'None'))
        
        logger.info(f"✅ Data reloaded: {len(PROJECTS_DATA)} projects, {contractors_count} contractors")
        
        return {
            "success": True,
            "message": "Data reloaded successfully",
            "projects_loaded": len(PROJECTS_DATA),
            "contractors_found": contractors_count,
            "file_path": path,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to reload data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load data: {str(e)}"
        )

@app.post("/api/v1/data/upload", tags=["Data Management"])
async def upload_excel_file(file: UploadFile = File(...)):
    """Upload and load data from Excel file"""
    global PROJECTS_DATA
    
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload Excel file (.xlsx or .xls)"
        )
    
    file_path = f"uploaded_{file.filename}"
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        new_data = load_excel_data(file_path)
        PROJECTS_DATA = new_data
        
        logger.info(f"✅ File uploaded and loaded: {len(PROJECTS_DATA)} projects")
        
        return {
            "success": True,
            "message": "File uploaded and data loaded successfully",
            "filename": file.filename,
            "projects_loaded": len(PROJECTS_DATA),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process uploaded file: {str(e)}"
        )

# ============================================================================
# SECTION 1: PROJECT SUMMARY DASHBOARD (Top KPI Cards)
# ============================================================================

@app.get(
    "/api/v1/dashboard/summary",
    response_model=ProjectSummaryKPI,
    tags=["Dashboard - Summary KPIs"],
    summary="Get Top-Level Dashboard KPIs"
)
async def get_dashboard_summary(
    bu: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    contractor: Optional[str] = Query(None),
    location: Optional[str] = Query(None)
):
    """
    **Dashboard Section 1: Project Summary KPIs**
    
    Returns all top-level KPIs for summary cards
    """
    
    filtered_data = filter_projects(
        PROJECTS_DATA,
        bu=bu,
        status=status,
        contractor=contractor,
        location=location
    )
    
    if not filtered_data:
        return ProjectSummaryKPI(
            total_projects=0,
            total_value_rm=0,
            total_completed_value=0,
            average_completion_percentage=0,
            total_quantity_installed=0,
            active_contractors_count=0,
            completed_projects=0,
            ongoing_projects=0,
            pending_projects=0,
            completion_vs_remaining={"completed": 0, "remaining": 100},
            total_variation_orders=0
        )
    
    total_projects = len(filtered_data)
    total_value = sum(p.get('value_rm', 0) or 0 for p in filtered_data)
    total_completed = sum(p.get('value_completed', 0) or 0 for p in filtered_data)
    total_vo = sum(p.get('vo', 0) or 0 for p in filtered_data)
    avg_completion = sum(p.get('percent', 0) or 0 for p in filtered_data) / total_projects
    total_quantity = sum(p.get('quantity', 0) or 0 for p in filtered_data if p.get('quantity'))
    
    contractors = set(p.get('contractor') for p in filtered_data if p.get('contractor') and p.get('contractor') != 'None')
    
    completed = len([p for p in filtered_data if p.get('status') == 'Completed'])
    ongoing = len([p for p in filtered_data if p.get('status') == 'On-going'])
    pending = len([p for p in filtered_data if p.get('status') == 'Pending'])
    
    return ProjectSummaryKPI(
        total_projects=total_projects,
        total_value_rm=round(total_value, 2),
        total_completed_value=round(total_completed, 2),
        average_completion_percentage=round(avg_completion, 2),
        total_quantity_installed=round(total_quantity, 2),
        active_contractors_count=len(contractors),
        completed_projects=completed,
        ongoing_projects=ongoing,
        pending_projects=pending,
        completion_vs_remaining={
            "completed": round(avg_completion, 2),
            "remaining": round(100 - avg_completion, 2)
        },
        total_variation_orders=round(total_vo, 2)
    )

# ============================================================================
# SECTION 2: PROJECT PROGRESS DASHBOARD
# ============================================================================

@app.get(
    "/api/v1/projects/progress",
    response_model=List[ProjectProgress],
    tags=["Dashboard - Project Progress"],
    summary="Get Project Progress for Visualization"
)
async def get_project_progress(
    bu: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    contractor: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    min_progress: Optional[float] = Query(None, ge=0, le=100),
    max_progress: Optional[float] = Query(None, ge=0, le=100)
):
    """
    **Dashboard Section 2: Project Progress Dashboard**
    
    Visual 1: Progress by Project (Horizontal Bar Chart)
    - Shows each project with progress bars
    - Color coding: Green (100%), Blue (<100%)
    - Includes contractor from 'Award' column, value, and status
    """
    
    filtered_data = filter_projects(
        PROJECTS_DATA,
        bu=bu,
        status=status,
        contractor=contractor,
        location=location,
        min_progress=min_progress,
        max_progress=max_progress
    )
    
    progress_data = []
    for project in filtered_data:
        progress_pct = project.get('percent', 0) or 0
        contractor_name = project.get('contractor', 'Unknown') or 'Unknown'
        
        progress_data.append(ProjectProgress(
            id=project.get('id', 0),
            project=project.get('project', 'Unknown'),
            progress_percent=round(progress_pct, 2),
            value_rm=round(project.get('value_rm', 0) or 0, 2),
            value_completed=round(project.get('value_completed', 0) or 0, 2),
            contractor=contractor_name,
            award=contractor_name,  # Also provide as 'award' for frontend compatibility
            status=project.get('status', 'Unknown') or 'Unknown',
            color_code="green" if progress_pct >= 100 else "blue",
            bu=project.get('bu', 'Unknown'),
            location=project.get('location', 'Unknown')
        ))
    
    # Sort by progress descending
    progress_data.sort(key=lambda x: x.progress_percent, reverse=True)
    
    return progress_data

# ============================================================================
# SECTION 3: COST DASHBOARD (Finance Section)
# ============================================================================

@app.get(
    "/api/v1/cost/analysis",
    response_model=List[CostAnalysis],
    tags=["Dashboard - Cost Analysis"],
    summary="Get Cost Analysis Data"
)
async def get_cost_analysis(
    bu: Optional[str] = Query(None),
    contractor: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """
    **Dashboard Section 3: Cost Dashboard**
    
    Visual 2: Budget vs Completed Value
    """
    
    filtered_data = filter_projects(
        PROJECTS_DATA,
        bu=bu,
        contractor=contractor,
        status=status
    )
    
    cost_data = []
    for project in filtered_data:
        value_rm = project.get('value_rm', 0) or 0
        value_completed = project.get('value_completed', 0) or 0
        vo = project.get('vo', 0) or 0
        
        remaining = max(0, value_rm - value_completed)
        efficiency = safe_divide(value_completed, value_rm) * 100
        
        cost_data.append(CostAnalysis(
            id=project.get('id', 0),
            project=project.get('project', 'Unknown'),
            bu=project.get('bu', 'Unknown'),
            value_rm=round(value_rm, 2),
            value_completed=round(value_completed, 2),
            variation_order=round(vo, 2),
            remaining_value=round(remaining, 2),
            cost_efficiency=round(efficiency, 2),
            contractor=project.get('contractor', 'Unknown') or 'Unknown',
            status=project.get('status', 'Unknown')
        ))
    
    return cost_data

@app.get(
    "/api/v1/cost/by-contractor",
    tags=["Dashboard - Cost Analysis"],
    summary="Get Cost Distribution by Contractor (Pie Chart)"
)
async def get_cost_by_contractor(
    bu: Optional[str] = Query(None)
):
    """
    **Dashboard Section 3: Cost Dashboard**
    
    Visual 3: Cost by Contractor (Pie/Donut Chart)
    - Shows total RM per contractor from 'Award' column
    - Identifies dependencies
    """
    
    filtered_data = filter_projects(PROJECTS_DATA, bu=bu)
    
    contractor_costs = defaultdict(float)
    for project in filtered_data:
        contractor = project.get('contractor', 'Unknown') or 'Unknown'
        if contractor != 'Unknown':
            value = project.get('value_rm', 0) or 0
            contractor_costs[contractor] += value
    
    # Sort by value descending
    sorted_costs = dict(sorted(contractor_costs.items(), key=lambda x: x[1], reverse=True))
    
    return {
        "data": {k: round(v, 2) for k, v in sorted_costs.items()},
        "total": round(sum(sorted_costs.values()), 2),
        "contractor_count": len(sorted_costs)
    }

@app.get(
    "/api/v1/cost/average-per-unit",
    tags=["Dashboard - Cost Analysis"],
    summary="Get Average Cost per Unit Comparison"
)
async def get_average_cost_per_unit(
    unit: Optional[str] = Query(None),
    contractor: Optional[str] = Query(None)
):
    """
    **Dashboard Section 3: Cost Dashboard**
    
    Visual 4: Average Cost per Unit (RM/unit)
    """
    
    filtered_data = filter_projects(PROJECTS_DATA, unit=unit, contractor=contractor)
    
    unit_data = defaultdict(lambda: {'total_cost': 0, 'total_quantity': 0, 'count': 0, 'contractors': set()})
    
    for project in filtered_data:
        unit_type = project.get('unit')
        if unit_type and project.get('average_rm_unit'):
            unit_data[unit_type]['total_cost'] += project.get('value_rm', 0) or 0
            unit_data[unit_type]['total_quantity'] += project.get('quantity', 0) or 0
            unit_data[unit_type]['count'] += 1
            if project.get('contractor'):
                unit_data[unit_type]['contractors'].add(project.get('contractor'))
    
    result = []
    for unit_type, data in unit_data.items():
        avg_cost = safe_divide(data['total_cost'], data['total_quantity'])
        result.append({
            'unit': unit_type,
            'average_cost_per_unit': avg_cost,
            'total_quantity': round(data['total_quantity'], 2),
            'total_value': round(data['total_cost'], 2),
            'project_count': data['count'],
            'contractors': list(data['contractors'])
        })
    
    # Sort by average cost descending
    result.sort(key=lambda x: x['average_cost_per_unit'], reverse=True)
    
    return result

# ============================================================================
# SECTION 4: QUANTITY & UNIT ANALYSIS DASHBOARD
# ============================================================================

@app.get(
    "/api/v1/quantity/analysis",
    response_model=List[UnitAnalysis],
    tags=["Dashboard - Quantity Analysis"],
    summary="Get Quantity Analysis by Unit Type"
)
async def get_quantity_analysis(
    bu: Optional[str] = Query(None)
):
    """
    **Dashboard Section 4: Quantity & Unit Analysis**
    
    Visual 5: Quantity of Work Done (Bar Chart)
    """
    
    filtered_data = filter_projects(PROJECTS_DATA, bu=bu)
    
    unit_analysis = defaultdict(lambda: {
        'total_quantity': 0,
        'total_value': 0,
        'count': 0,
        'costs': []
    })
    
    for project in filtered_data:
        unit = project.get('unit')
        if unit:
            unit_analysis[unit]['total_quantity'] += project.get('quantity', 0) or 0
            unit_analysis[unit]['total_value'] += project.get('value_rm', 0) or 0
            unit_analysis[unit]['count'] += 1
            if project.get('average_rm_unit'):
                unit_analysis[unit]['costs'].append(project.get('average_rm_unit'))
    
    result = []
    for unit, data in unit_analysis.items():
        avg_cost = safe_divide(data['total_value'], data['total_quantity'])
        min_cost = min(data['costs']) if data['costs'] else 0
        max_cost = max(data['costs']) if data['costs'] else 0
        
        result.append(UnitAnalysis(
            unit=unit,
            total_quantity=round(data['total_quantity'], 2),
            total_value=round(data['total_value'], 2),
            average_cost_per_unit=avg_cost,
            project_count=data['count'],
            min_cost_per_unit=round(min_cost, 2),
            max_cost_per_unit=round(max_cost, 2)
        ))
    
    # Sort by total value descending
    result.sort(key=lambda x: x.total_value, reverse=True)
    
    return result

@app.get(
    "/api/v1/quantity/bubble-chart-data",
    response_model=List[BubbleChartData],
    tags=["Dashboard - Quantity Analysis"],
    summary="Get Bubble Chart Data for Cost per Unit"
)
async def get_bubble_chart_data():
    """
    **Dashboard Section 4: Quantity & Unit Analysis**
    
    Visual 6: Cost per Unit Comparison (Bubble Chart)
    """
    
    unit_data = defaultdict(lambda: {
        'projects': [],
        'total_value': 0,
        'avg_costs': [],
        'count': 0
    })
    
    for project in PROJECTS_DATA:
        unit = project.get('unit')
        if unit and project.get('average_rm_unit'):
            unit_data[unit]['projects'].append(project.get('project'))
            unit_data[unit]['total_value'] += project.get('value_rm', 0) or 0
            unit_data[unit]['avg_costs'].append(project.get('average_rm_unit', 0) or 0)
            unit_data[unit]['count'] += 1
    
    result = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for idx, (unit, data) in enumerate(unit_data.items()):
        avg_cost = sum(data['avg_costs']) / len(data['avg_costs']) if data['avg_costs'] else 0
        
        result.append(BubbleChartData(
            unit=unit,
            bubble_size=round(data['total_value'], 2),
            average_cost_per_unit=round(avg_cost, 2),
            project_count=data['count'],
            total_value=round(data['total_value'], 2),
            color_category=colors[idx % len(colors)]
        ))
    
    # Sort by bubble size descending
    result.sort(key=lambda x: x.bubble_size, reverse=True)
    
    return result

# ============================================================================
# SECTION 5: CDS TRACKING DASHBOARD
# ============================================================================

@app.get(
    "/api/v1/cds/tracking",
    response_model=List[CDSTracking],
    tags=["Dashboard - CDS Tracking"],
    summary="Get CDS Tracking Data"
)
async def get_cds_tracking(
    status: Optional[str] = Query(None),
    bu: Optional[str] = Query(None)
):
    """
    **Dashboard Section 5: CDS Tracking**
    
    Visual 7: CDS vs Progress (Line/Bar Chart)
    """
    
    filtered_data = filter_projects(PROJECTS_DATA, status=status, bu=bu)
    
    cds_data = []
    for project in filtered_data:
        cds = project.get('cds')
        if cds and cds != 'None':
            cds_data.append(CDSTracking(
                id=project.get('id', 0),
                cds=cds,
                project=project.get('project', 'Unknown'),
                bu=project.get('bu', 'Unknown'),
                estimated_days=int(project.get('estimated_days', 0) or 0),
                finish_date=project.get('finish_date'),
                status=project.get('status', 'Unknown'),
                completion_percent=round(project.get('percent', 0) or 0, 2),
                contractor=project.get('contractor', 'Unknown') or 'Unknown',
                days_elapsed=None
            ))
    
    # Sort by completion percent descending
    cds_data.sort(key=lambda x: x.completion_percent, reverse=True)
    
    return cds_data

@app.get(
    "/api/v1/cds/timeline",
    response_model=List[TimelineData],
    tags=["Dashboard - CDS Tracking"],
    summary="Get Timeline Data for Gantt Chart"
)
async def get_timeline_data(
    contractor: Optional[str] = Query(None),
    bu: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """
    **Dashboard Section 5: CDS Tracking**
    
    Visual 8: Planned vs Actual Finish Date (Gantt Chart)
    """
    
    filtered_data = filter_projects(PROJECTS_DATA, contractor=contractor, bu=bu, status=status)
    
    timeline_data = []
    for project in filtered_data:
        timeline_data.append(TimelineData(
            id=project.get('id', 0),
            project=project.get('project', 'Unknown'),
            contractor=project.get('contractor', 'Unknown') or 'Unknown',
            bu=project.get('bu', 'Unknown'),
            planned_finish_date=project.get('finish_date'),
            estimated_days=project.get('estimated_days'),
            status=project.get('status', 'Unknown'),
            completion_percent=round(project.get('percent', 0) or 0, 2),
            start_date=None,
            delay_days=None
        ))
    
    # Sort by planned finish date
    timeline_data.sort(key=lambda x: x.planned_finish_date or '9999-12-31')
    
    return timeline_data

# ============================================================================
# SECTION 6: CONTRACTOR PERFORMANCE DASHBOARD
# ============================================================================

@app.get(
    "/api/v1/contractors/performance",
    response_model=List[ContractorKPI],
    tags=["Dashboard - Contractor Performance"],
    summary="Get Contractor Performance KPIs"
)
async def get_contractor_performance(
    min_projects: Optional[int] = Query(None, ge=1),
    bu: Optional[str] = Query(None)
):
    """
    **Dashboard Section 6: Contractor Performance**
    
    Visual 9: Contractor-wise KPIs (Table/Bar Chart)
    - Uses 'Award' column data
    """
    
    filtered_data = filter_projects(PROJECTS_DATA, bu=bu)
    
    contractor_data = defaultdict(lambda: {
        'total_value': 0,
        'total_completed': 0,
        'total_cost_per_unit': 0,
        'total_completion': 0,
        'project_count': 0,
        'total_quantity': 0,
        'unit_count': 0,
        'ongoing': 0,
        'completed': 0
    })
    
    for project in filtered_data:
        contractor = project.get('contractor', 'Unknown') or 'Unknown'
        if contractor != 'Unknown':
            data = contractor_data[contractor]
            data['total_value'] += project.get('value_rm', 0) or 0
            data['total_completed'] += project.get('value_completed', 0) or 0
            data['total_completion'] += project.get('percent', 0) or 0
            data['project_count'] += 1
            data['total_quantity'] += project.get('quantity', 0) or 0
            
            if project.get('average_rm_unit'):
                data['total_cost_per_unit'] += project.get('average_rm_unit', 0) or 0
                data['unit_count'] += 1
            
            if project.get('status') == 'On-going':
                data['ongoing'] += 1
            elif project.get('status') == 'Completed':
                data['completed'] += 1
    
    result = []
    for contractor, data in contractor_data.items():
        if min_projects and data['project_count'] < min_projects:
            continue
        
        avg_cost = safe_divide(data['total_cost_per_unit'], data['unit_count'])
        avg_completion = safe_divide(data['total_completion'], data['project_count'])
        efficiency = safe_divide(data['total_completed'], data['total_value']) * 100
        
        result.append(ContractorKPI(
            contractor=contractor,
            total_value=round(data['total_value'], 2),
            total_completed_value=round(data['total_completed'], 2),
            average_cost_per_unit=avg_cost,
            completion_percentage=avg_completion,
            number_of_projects=data['project_count'],
            total_quantity=round(data['total_quantity'], 2),
            ongoing_projects=data['ongoing'],
            completed_projects=data['completed'],
            efficiency_score=round(efficiency, 2)
        ))
    
    # Sort by total value descending
    result.sort(key=lambda x: x.total_value, reverse=True)
    
    return result

@app.get(
    "/api/v1/contractors/workload",
    tags=["Dashboard - Contractor Performance"],
    summary="Get Contractor Workload Distribution"
)
async def get_contractor_workload():
    """
    **Dashboard Section 6: Contractor Performance**
    
    Visual 10: Contractor Workload (Pie/Bar Chart)
    """
    
    workload = defaultdict(lambda: {'count': 0, 'ongoing': 0, 'completed': 0, 'pending': 0})
    
    for project in PROJECTS_DATA:
        contractor = project.get('contractor', 'Unknown') or 'Unknown'
        if contractor != 'Unknown':
            workload[contractor]['count'] += 1
            status = project.get('status')
            if status == 'On-going':
                workload[contractor]['ongoing'] += 1
            elif status == 'Completed':
                workload[contractor]['completed'] += 1
            elif status == 'Pending':
                workload[contractor]['pending'] += 1
    
    # Sort by count descending
    sorted_workload = dict(sorted(workload.items(), key=lambda x: x[1]['count'], reverse=True))
    
    return {
        "data": sorted_workload,
        "total_contractors": len(sorted_workload),
        "total_projects": sum(w['count'] for w in sorted_workload.values())
    }

# ============================================================================
# SECTION 7: BUSINESS UNIT ANALYSIS DASHBOARD
# ============================================================================

@app.get(
    "/api/v1/business-units/analysis",
    response_model=List[BUAnalysis],
    tags=["Dashboard - Business Unit Analysis"],
    summary="Get Business Unit Analysis"
)
async def get_bu_analysis():
    """
    **Dashboard Section 7: Business Unit Dashboard**
    
    Visual 11: BU-wise Cost (Bar Chart)
    """
    
    bu_data = defaultdict(lambda: {
        'total_value': 0,
        'completed_value': 0,
        'total_vo': 0,
        'project_count': 0,
        'total_completion': 0,
        'completed': 0,
        'ongoing': 0,
        'pending': 0
    })
    
    for project in PROJECTS_DATA:
        bu = project.get('bu', 'Unknown')
        data = bu_data[bu]
        data['total_value'] += project.get('value_rm', 0) or 0
        data['completed_value'] += project.get('value_completed', 0) or 0
        data['total_vo'] += project.get('vo', 0) or 0
        data['project_count'] += 1
        data['total_completion'] += project.get('percent', 0) or 0
        
        status = project.get('status')
        if status == 'Completed':
            data['completed'] += 1
        elif status == 'On-going':
            data['ongoing'] += 1
        elif status == 'Pending':
            data['pending'] += 1
    
    result = []
    for bu, data in bu_data.items():
        avg_completion = safe_divide(data['total_completion'], data['project_count'])
        efficiency = safe_divide(data['completed_value'], data['total_value']) * 100
        
        result.append(BUAnalysis(
            bu=bu,
            total_value=round(data['total_value'], 2),
            total_completed_value=round(data['completed_value'], 2),
            total_vo=round(data['total_vo'], 2),
            project_count=data['project_count'],
            average_completion=avg_completion,
            completed_projects=data['completed'],
            ongoing_projects=data['ongoing'],
            pending_projects=data['pending'],
            efficiency=round(efficiency, 2)
        ))
    
    # Sort by total value descending
    result.sort(key=lambda x: x.total_value, reverse=True)
    
    return result

# ============================================================================
# SECTION 8: LOCATION ANALYSIS DASHBOARD
# ============================================================================

@app.get(
    "/api/v1/locations/analysis",
    response_model=List[LocationAnalysis],
    tags=["Dashboard - Location Analysis"],
    summary="Get Location-wise Analysis"
)
async def get_location_analysis(
    bu: Optional[str] = Query(None)
):
    """
    **Dashboard Section 8: Location Dashboard**
    
    Visual 12: Location-wise Project Count
    """
    
    filtered_data = filter_projects(PROJECTS_DATA, bu=bu)
    
    location_data = defaultdict(lambda: {
        'bu': None,
        'project_count': 0,
        'total_value': 0,
        'completed_value': 0,
        'contractors': set(),
        'total_completion': 0,
        'status_dist': defaultdict(int)
    })
    
    for project in filtered_data:
        location = project.get('location', 'Unknown')
        data = location_data[location]
        
        if not data['bu']:
            data['bu'] = project.get('bu', 'Unknown')
        
        data['project_count'] += 1
        data['total_value'] += project.get('value_rm', 0) or 0
        data['completed_value'] += project.get('value_completed', 0) or 0
        data['total_completion'] += project.get('percent', 0) or 0
        
        if project.get('contractor') and project.get('contractor') != 'Unknown':
            data['contractors'].add(project.get('contractor'))
        
        status = project.get('status', 'Unknown')
        data['status_dist'][status] += 1
    
    result = []
    for location, data in location_data.items():
        avg_completion = safe_divide(data['total_completion'], data['project_count'])
        
        result.append(LocationAnalysis(
            location=location,
            bu=data['bu'],
            project_count=data['project_count'],
            total_value=round(data['total_value'], 2),
            total_completed_value=round(data['completed_value'], 2),
            contractors=list(data['contractors']),
            average_completion=avg_completion,
            status_distribution=dict(data['status_dist'])
        ))
    
    # Sort by total value descending
    result.sort(key=lambda x: x.total_value, reverse=True)
    
    return result

# ============================================================================
# SECTION 9: STAFF WORKLOAD DASHBOARD
# ============================================================================

@app.get(
    "/api/v1/staff/workload",
    response_model=List[StaffWorkload],
    tags=["Dashboard - Staff Workload"],
    summary="Get Staff Workload Analysis"
)
async def get_staff_workload():
    """
    **Dashboard Section 9: Staff Workload Dashboard**
    
    Staff supervision analysis
    """
    
    staff_columns = ['safuan', 'hamdee', 'woradate', 'nantawat', 'alya', 'vignes', 'azim']
    
    staff_data = defaultdict(lambda: {
        'total_value': 0,
        'completed_value': 0,
        'project_count': 0,
        'total_completion': 0,
        'projects': []
    })
    
    for project in PROJECTS_DATA:
        for staff in staff_columns:
            value = project.get(staff)
            if value and value > 0:
                staff_data[staff]['total_value'] += value
                staff_data[staff]['project_count'] += 1
                staff_data[staff]['total_completion'] += project.get('percent', 0) or 0
                staff_data[staff]['projects'].append(project.get('project', 'Unknown'))
                
                # Proportional completed value based on staff's share
                project_value = project.get('value_rm', 0) or 0
                if project_value > 0:
                    staff_share = value / project_value
                    completed = (project.get('value_completed', 0) or 0) * staff_share
                    staff_data[staff]['completed_value'] += completed
    
    result = []
    for staff, data in staff_data.items():
        if data['project_count'] > 0:
            avg_completion = safe_divide(data['total_completion'], data['project_count'])
            efficiency = safe_divide(data['completed_value'], data['total_value']) * 100
            
            result.append(StaffWorkload(
                staff_name=staff.capitalize(),
                total_value_supervised=round(data['total_value'], 2),
                projects_under_supervision=data['project_count'],
                average_completion=avg_completion,
                total_completed_value=round(data['completed_value'], 2),
                efficiency=round(efficiency, 2),
                project_list=data['projects'][:5]
            ))
    
    # Sort by total value descending
    result.sort(key=lambda x: x.total_value_supervised, reverse=True)
    
    return result

# ============================================================================
# SECTION 10: PROJECT DETAILS & DRILL-DOWN
# ============================================================================

@app.get(
    "/api/v1/projects",
    response_model=List[Project],
    tags=["Projects - CRUD Operations"],
    summary="Get All Projects with Advanced Filtering"
)
async def get_all_projects(
    bu: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    contractor: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    unit: Optional[str] = Query(None),
    cds: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    **Get All Projects with Advanced Filtering**
    """
    
    filtered_data = filter_projects(
        PROJECTS_DATA,
        bu=bu,
        status=status,
        contractor=contractor,
        location=location,
        unit=unit,
        cds=cds
    )
    
    # Apply pagination
    paginated_data = filtered_data[skip:skip + limit]
    
    result = []
    for p in paginated_data:
        result.append(Project(
            id=p.get('id', 0),
            project=p.get('project', ''),
            bu=p.get('bu', ''),
            location=p.get('location', ''),
            status=p.get('status', ''),
            percent=p.get('percent', 0) or 0,
            value_rm=p.get('value_rm', 0) or 0,
            value_completed=p.get('value_completed', 0) or 0,
            vo=p.get('vo', 0) or 0,
            cds=p.get('cds'),
            estimated_days=p.get('estimated_days'),
            finish_date=p.get('finish_date'),
            unit=p.get('unit'),
            quantity=p.get('quantity'),
            average_rm_unit=p.get('average_rm_unit'),
            contractor=p.get('contractor'),
            safuan=p.get('safuan'),
            hamdee=p.get('hamdee'),
            woradate=p.get('woradate'),
            nantawat=p.get('nantawat'),
            alya=p.get('alya'),
            vignes=p.get('vignes'),
            azim=p.get('azim')
        ))
    
    return result

@app.get(
    "/api/v1/projects/{project_id}",
    response_model=ProjectDetail,
    tags=["Projects - CRUD Operations"],
    summary="Get Detailed Project Information (Drill-Down)"
)
async def get_project_detail(project_id: int):
    """
    **Dashboard Section 10: Detailed Project Page (Drill-Down)**
    """
    
    project = next((p for p in PROJECTS_DATA if p.get('id') == project_id), None)
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID {project_id} not found"
        )
    
    # Related contractors (same location or BU)
    related = [p.get('contractor') for p in PROJECTS_DATA 
               if (p.get('location') == project.get('location') or 
                   p.get('bu') == project.get('bu')) 
               and p.get('contractor') and p.get('id') != project_id]
    related_contractors = list(set(filter(None, related)))
    
    # Cost breakdown
    value_rm = project.get('value_rm', 0) or 0
    value_completed = project.get('value_completed', 0) or 0
    vo = project.get('vo', 0) or 0
    
    cost_breakdown = {
        'total_value': round(value_rm, 2),
        'completed_value': round(value_completed, 2),
        'remaining_value': round(max(0, value_rm - value_completed), 2),
        'variation_order': round(vo, 2),
        'adjusted_total': round(value_rm + vo, 2)
    }
    
    # Timeline info
    timeline_info = {
        'estimated_days': project.get('estimated_days'),
        'finish_date': project.get('finish_date'),
        'cds': project.get('cds'),
        'status': project.get('status')
    }
    
    # Staff assignments
    staff_assignments = {}
    staff_columns = ['safuan', 'hamdee', 'woradate', 'nantawat', 'alya', 'vignes', 'azim']
    for staff in staff_columns:
        value = project.get(staff)
        if value and value > 0:
            staff_assignments[staff.capitalize()] = round(value, 2)
    
    # Performance metrics
    completion_pct = project.get('percent', 0) or 0
    performance_metrics = {
        'completion_percentage': round(completion_pct, 2),
        'cost_efficiency': round(safe_divide(value_completed, value_rm) * 100, 2),
        'average_cost_per_unit': round(project.get('average_rm_unit', 0) or 0, 2),
        'quantity': round(project.get('quantity', 0) or 0, 2),
        'unit': project.get('unit')
    }
    
    project_model = Project(
        id=project.get('id', 0),
        project=project.get('project', ''),
        bu=project.get('bu', ''),
        location=project.get('location', ''),
        status=project.get('status', ''),
        percent=completion_pct,
        value_rm=value_rm,
        value_completed=value_completed,
        vo=vo,
        cds=project.get('cds'),
        estimated_days=project.get('estimated_days'),
        finish_date=project.get('finish_date'),
        unit=project.get('unit'),
        quantity=project.get('quantity'),
        average_rm_unit=project.get('average_rm_unit'),
        contractor=project.get('contractor'),
        safuan=project.get('safuan'),
        hamdee=project.get('hamdee'),
        woradate=project.get('woradate'),
        nantawat=project.get('nantawat'),
        alya=project.get('alya'),
        vignes=project.get('vignes'),
        azim=project.get('azim')
    )
    
    return ProjectDetail(
        project=project_model,
        related_contractors=related_contractors,
        cost_breakdown=cost_breakdown,
        timeline_info=timeline_info,
        staff_assignments=staff_assignments,
        performance_metrics=performance_metrics
    )

# ============================================================================
# SECTION 11: FILTER OPTIONS (Slicers)
# ============================================================================

@app.get(
    "/api/v1/filters/options",
    response_model=FilterOptions,
    tags=["Dashboard - Filters"],
    summary="Get All Filter Options for Dashboard Slicers"
)
async def get_filter_options():
    """
    **Dashboard Section 11: Filters (Slicers)**
    
    All available filter options including contractors from 'Award' column
    """
    
    projects = sorted(set(p.get('project') for p in PROJECTS_DATA if p.get('project') and p.get('project') != 'None'))
    contractors = sorted(set(p.get('contractor') for p in PROJECTS_DATA if p.get('contractor') and p.get('contractor') != 'None'))
    bus = sorted(set(p.get('bu') for p in PROJECTS_DATA if p.get('bu')))
    statuses = sorted(set(p.get('status') for p in PROJECTS_DATA if p.get('status')))
    locations = sorted(set(p.get('location') for p in PROJECTS_DATA if p.get('location')))
    units = sorted(set(p.get('unit') for p in PROJECTS_DATA if p.get('unit') and p.get('unit') != 'None'))
    cds_list = sorted(set(p.get('cds') for p in PROJECTS_DATA if p.get('cds') and p.get('cds') != 'None'))
    
    # Extract months and years from finish_date
    months = set()
    years = set()
    for p in PROJECTS_DATA:
        finish_date = p.get('finish_date')
        if finish_date and finish_date != 'None':
            try:
                if '/' in finish_date:
                    parts = finish_date.split('/')
                    if len(parts) == 3:
                        month = parts[0]
                        year = parts[2]
                        months.add(month)
                        years.add(int(year))
                elif '-' in finish_date:
                    date_obj = datetime.fromisoformat(finish_date)
                    months.add(str(date_obj.month))
                    years.add(date_obj.year)
            except:
                pass
    
    return FilterOptions(
        projects=projects,
        contractors=contractors,
        bus=bus,
        statuses=statuses,
        locations=locations,
        units=units,
        cds_list=cds_list,
        months=sorted(months, key=lambda x: int(x)) if months else [],
        years=sorted(list(years)) if years else []
    )

# ============================================================================
# ADDITIONAL UTILITY ENDPOINTS
# ============================================================================

@app.get(
    "/api/v1/statistics/overall",
    tags=["Dashboard - Statistics"],
    summary="Get Overall Dashboard Statistics"
)
async def get_overall_statistics():
    """
    Comprehensive statistics for executive summary
    """
    
    if not PROJECTS_DATA:
        return {
            "message": "No data available",
            "total_projects": 0
        }
    
    total_projects = len(PROJECTS_DATA)
    total_value = sum(p.get('value_rm', 0) or 0 for p in PROJECTS_DATA)
    total_completed = sum(p.get('value_completed', 0) or 0 for p in PROJECTS_DATA)
    total_vo = sum(p.get('vo', 0) or 0 for p in PROJECTS_DATA)
    
    contractors = set(p.get('contractor') for p in PROJECTS_DATA if p.get('contractor') and p.get('contractor') != 'None')
    locations = set(p.get('location') for p in PROJECTS_DATA if p.get('location'))
    bus = set(p.get('bu') for p in PROJECTS_DATA if p.get('bu'))
    
    status_counts = defaultdict(int)
    for p in PROJECTS_DATA:
        status_counts[p.get('status', 'Unknown')] += 1
    
    avg_completion = sum(p.get('percent', 0) or 0 for p in PROJECTS_DATA) / total_projects
    completion_efficiency = safe_divide(total_completed, total_value) * 100
    
    return {
        "total_projects": total_projects,
        "total_value_rm": round(total_value, 2),
        "total_completed_value_rm": round(total_completed, 2),
        "total_variation_orders_rm": round(total_vo, 2),
        "adjusted_total_value": round(total_value + total_vo, 2),
        "unique_contractors": len(contractors),
        "unique_locations": len(locations),
        "unique_business_units": len(bus),
        "status_distribution": dict(status_counts),
        "average_completion_percentage": round(avg_completion, 2),
        "completion_efficiency": round(completion_efficiency, 2),
        "data_quality": {
            "projects_with_cds": len([p for p in PROJECTS_DATA if p.get('cds') and p.get('cds') != 'None']),
            "projects_with_contractors": len([p for p in PROJECTS_DATA if p.get('contractor') and p.get('contractor') != 'None']),
            "projects_with_finish_date": len([p for p in PROJECTS_DATA if p.get('finish_date') and p.get('finish_date') != 'None']),
        }
    }

@app.get(
    "/api/v1/export/csv",
    tags=["Data Export"],
    summary="Export Filtered Data as CSV"
)
async def export_csv(
    bu: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    contractor: Optional[str] = Query(None),
    location: Optional[str] = Query(None)
):
    """
    Export filtered data as CSV for external analysis
    """
    
    filtered_data = filter_projects(
        PROJECTS_DATA,
        bu=bu,
        status=status,
        contractor=contractor,
        location=location
    )
    
    if not filtered_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No data found matching the filters"
        )
    
    df = pd.DataFrame(filtered_data)
    csv_data = df.to_csv(index=False)
    
    return {
        "filename": f"projects_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "data": csv_data,
        "record_count": len(filtered_data),
        "filters_applied": {
            "bu": bu,
            "status": status,
            "contractor": contractor,
            "location": location
        }
    }

# ============================================================================
# ROOT & HEALTH CHECK
# ============================================================================

@app.get("/api", tags=["Root"])
async def api_info():
    """API health check and information"""
    return {
        "message": "Construction Project Dashboard API - FIXED",
        "version": "2.0.1",
        "status": "active",
        "data_source": "Excel" if os.path.exists(EXCEL_FILE_PATH) else "Sample Data",
        "projects_loaded": len(PROJECTS_DATA),
        "contractors_loaded": len(set(p.get('contractor') for p in PROJECTS_DATA if p.get('contractor') and p.get('contractor') != 'None')),
        "last_updated": datetime.now().isoformat(),
        "fix_notes": "Now correctly reads 'Award' column from Excel as contractor data",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "summary": "/api/v1/dashboard/summary",
            "progress": "/api/v1/projects/progress",
            "cost": "/api/v1/cost/analysis",
            "contractors": "/api/v1/contractors/performance",
            "staff": "/api/v1/staff/workload",
            "filters": "/api/v1/filters/options",
            "reload": "/api/v1/data/reload",
            "upload": "/api/v1/data/upload"
        }
    }

@app.get("/health", tags=["Root"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": len(PROJECTS_DATA) > 0,
        "projects_count": len(PROJECTS_DATA),
        "contractors_count": len(set(p.get('contractor') for p in PROJECTS_DATA if p.get('contractor') and p.get('contractor') != 'None')),
        "data_source": "Excel" if os.path.exists(EXCEL_FILE_PATH) else "Sample Data"
    }

# ============================================================================
# ERROR HANDLING
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled Exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "message": "An internal server error occurred",
            "detail": str(exc) if app.debug else "Contact support",
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("=" * 70)
    logger.info("🚀 Construction Dashboard API Starting...")
    logger.info("=" * 70)
    logger.info(f"📊 Data Source: {'Excel File' if os.path.exists(EXCEL_FILE_PATH) else 'Sample Data'}")
    logger.info(f"📁 Excel File: {EXCEL_FILE_PATH}")
    logger.info(f"✅ Projects Loaded: {len(PROJECTS_DATA)}")
    contractors_count = len(set(p.get('contractor') for p in PROJECTS_DATA if p.get('contractor') and p.get('contractor') != 'None'))
    logger.info(f"✅ Contractors Found: {contractors_count}")
    logger.info("=" * 70)
    logger.info("📖 API Documentation: http://localhost:8000/docs")
    logger.info("🔄 Reload Data: POST http://localhost:8000/api/v1/data/reload")
    logger.info("📤 Upload Data: POST http://localhost:8000/api/v1/data/upload")
    logger.info("=" * 70)

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("👋 Construction Dashboard API Shutting Down...")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🏗️  CONSTRUCTION PROJECT DASHBOARD API v2.0.1 - FIXED")
    print("=" * 70)
    print(f"📂 Excel file: {EXCEL_FILE_PATH}")
    print("✅ FIX: Now correctly reads 'Award' column as contractor data")
    print("💡 Place your Excel file as 'projects_data.xlsx' to load your data")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )