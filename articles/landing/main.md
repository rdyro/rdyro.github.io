<p align="center">
<h1>Landing an Airplane Autonomously</h1>
</p>

---

# Introduction

Outline:
- X-Plane 11 Simulator is a high-fidelity simulator for aircraft
- the simulator is black-box, the physics is not exposed, the world cannot be rolled back
    - difficulty of the environment
- designing a model-based controller, a sample efficient one, requires a model
    - types of models
    - linear model vs nonlinear model
    - learning controllable dynamics
- designing a controller
    - controller 1st attempt - Model Predictive Control
    - controller 2nd attempt - Linear Quadratic Regulator
- the controller doesn't work
    - tuning the controller - parametric search
- conclusions
  - sample complexity for learning a "policy" (RL perspective)
  - difficulty in having a single cost function for "far" and "near" the goal
    - applying constraints to the states

# Addressing the Black-Box Problem with the Simulator

# Learning the Dynamics

# Designing the Controller

# The Controller Doesn't Work - Tuning the Cost Parameters

# Conclusion & Future Work