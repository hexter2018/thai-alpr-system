"""Vehicle Master Data CRUD API"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from ...database import get_async_db
from ...models import MasterVehicle
from ...schemas import VehicleCreate, VehicleUpdate, VehicleResponse

router = APIRouter()

@router.get("/", response_model=List[VehicleResponse])
async def list_vehicles(skip: int = 0, limit: int = 50, db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(MasterVehicle).offset(skip).limit(limit))
    return result.scalars().all()

@router.post("/", response_model=VehicleResponse, status_code=201)
async def create_vehicle(vehicle: VehicleCreate, db: AsyncSession = Depends(get_async_db)):
    db_vehicle = MasterVehicle(**vehicle.dict())
    db.add(db_vehicle)
    await db.commit()
    await db.refresh(db_vehicle)
    return db_vehicle

@router.get("/{vehicle_id}", response_model=VehicleResponse)
async def get_vehicle(vehicle_id: int, db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(MasterVehicle).where(MasterVehicle.id == vehicle_id))
    vehicle = result.scalar_one_or_none()
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return vehicle

@router.put("/{vehicle_id}", response_model=VehicleResponse)
async def update_vehicle(vehicle_id: int, vehicle: VehicleUpdate, db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(MasterVehicle).where(MasterVehicle.id == vehicle_id))
    db_vehicle = result.scalar_one_or_none()
    if not db_vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    for key, value in vehicle.dict(exclude_unset=True).items():
        setattr(db_vehicle, key, value)
    await db.commit()
    await db.refresh(db_vehicle)
    return db_vehicle

@router.delete("/{vehicle_id}")
async def delete_vehicle(vehicle_id: int, db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(MasterVehicle).where(MasterVehicle.id == vehicle_id))
    vehicle = result.scalar_one_or_none()
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    await db.delete(vehicle)
    await db.commit()
    return {"message": "Vehicle deleted"}
