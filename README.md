# AI Food Detector

This project detects dishes in food photos using a Django backend and a Vite + React frontend.

## Quick start

Backend (Django):
- Create a virtual env and install requirements
- Run migrations: `python manage.py migrate`
- Start server: `python manage.py runserver`

Frontend (Vite + React):
- Install dependencies: `npm install` in `frontend/`
- Start dev server: `npm run dev`

## Notes
- The trained model (`backend/photo_processor/models/dish_detection.pth`) is tracked via Git LFS.
- See `backend/README.md` for backend-specific notes (if present).

---

Contributions welcome.