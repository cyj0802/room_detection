# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.jobs import router as jobs_router
from app.api.callbacks import router as callbacks_router

app = FastAPI(title="Room Detection Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)
app.include_router(callbacks_router)

@app.get("/")
async def root():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        reload=True, port=8000,
        reload_dirs=["app","."],
        reload_excludes=[".venv/*","**/site-packages/**","data/*","logs/*"],
    )
