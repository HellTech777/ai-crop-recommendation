from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRoute
from agri_app.your_model_code import recommend_crop, get_available_cities  # adjust if file is renamed

app = FastAPI()

# Mount static files (CSS, images, etc.)
app.mount("/static", StaticFiles(directory="agri_app/static"), name="static")

# Load templates
templates = Jinja2Templates(directory="agri_app/templates")

# Home page with form
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    cities = get_available_cities()
    print("Dropdown cities:", cities)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "cities": cities
    })

# Handle form submission
@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, city: str = Form(...), yield_q: float = Form(...)):
    result = recommend_crop(city, yield_q)

    # Handle error case from recommend_crop
    if "error" in result:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "result": {
                "crop": "N/A",
                "image": "placeholder.jpg",
                "city": city,
                "temp": "N/A",
                "humidity": "N/A",
                "soil": {},
                "price": "N/A",
                "profit": "N/A",
                "error": result["error"]
            }
        })

    # Pass full result to template
    return templates.TemplateResponse("result.html", {
        "request": request,
        "result": {
            "crop": result["crop"],
            "image": result["image"],
            "city": result["city"],
            "temp": result["temp"],
            "humidity": result["humidity"],
            "soil": {
                "n": result["soil"]["n"],
                "p": result["soil"]["p"],
                "k": result["soil"]["k"],
                "ph": result["soil"]["ph"],
                "rainfall": result["soil"]["rainfall"]
            },
            "price": result["price"],
            "profit": result["profit"]
        }
    })

# Patch to allow HEAD requests on GET routes (for health checks)
def allow_head_for_get(route: APIRoute):
    if "GET" in route.methods:
        route.methods.add("HEAD")

for route in app.routes:
    if isinstance(route, APIRoute):
        allow_head_for_get(route)
