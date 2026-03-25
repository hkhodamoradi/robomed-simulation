# RoboMed Simulation Pipeline

Webots-based simulation pipeline for a bedside medical assistant robot, forming the low-level robotics layer of the RoboMed project.

## Overview

This repository focuses on simulation, perception, and control of a medical assistant robot operating in a hospital-like environment.

It enables development and testing of robot behavior such as patient localization, safe navigation, and interaction with bedside equipment before integrating higher-level AI decision pipelines and dashboard workflows.

## Current Capabilities

- Patient localization in a simulated environment
- Safe approach to patient bed
- Bed label identification
- Monitor detection
- Monitor snapshot capture for downstream AI analysis

## Planned Extensions

- Unified controller architecture
- Agent-based QR / vitals extraction
- Human-in-the-loop alerting
- Emergency handling workflows

## Repository Structure

- `controllers/` robot control modules (navigation, perception, behaviors)
- `worlds/` Webots world files
- `assets/` monitor images and bed labels
- `docs/` architecture notes and documentation
- `videos/` demo recordings (optional)

## Setup

You can create the basic project structure using:

```bash
chmod +x setup_repo_structure.sh
./setup_repo_structure.sh

## Related Project

Main RoboMed system:
https://github.com/hkhodamoradi/robomed-ai

## Demo

Robot perspective:
https://youtu.be/BnZjWKFgz0M

Top view:
https://youtu.be/SilspwN7Z0g
