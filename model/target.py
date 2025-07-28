# model/target.py

from pydantic import BaseModel

class Target(BaseModel):
    index: int
    r: float           # radial distance from origin (m)
    theta: float       # angle in degrees (relative to vertical)
    tin: float         # glint spacing in µs
    NoG: int           # number of glints

