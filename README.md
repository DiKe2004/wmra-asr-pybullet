# WMRA ASR PyBullet

Source code for the undergraduate thesis:

**SIMULASI WHEELCHAIR-MOUNTED ROBOTIC ARM BERBASIS AUTOMATIC SPEECH RECOGNITION UNTUK PENGAMBILAN BARANG**

## Overview

This repository contains the source code used to develop and evaluate a
wheelchair-mounted robotic arm simulation in PyBullet.

The system integrates:

- Automatic Speech Recognition using Whisper small through faster-whisper
- Indonesian command parsing
- Object search and target matching
- Kinova JACO Gen 2 simulation
- PyBullet inverse kinematics
- Virtual grasp using fixed constraint
- Lift and handover processes
- Basket placement evaluation
- Contact monitoring
- P01 to P12 evaluation scripts

## System

The simulated system consists of:

- Virtual electric wheelchair platform
- Kinova JACO Gen 2 robotic arm
- Three-finger gripper
- Two racks with three shelf levels
- Target and distractor objects
- Synthetic PyBullet camera
- Handover area
- Basket

## Automatic Speech Recognition

The ASR module uses:

- Whisper model variant: small
- Implementation: faster-whisper
- Backend: CTranslate2
- Language: Indonesian (`id`)
- Beam size: 5
- VAD filter: False
- CUDA
- Compute type: float16

The ASR evaluation uses four paired audio conditions:

- Clean
- SNR 20 dB
- SNR 10 dB
- SNR 5 dB

## Evaluation

The system is evaluated using P01 to P12:

P01: Simulation environment validation  
P02: ASR evaluation  
P03: Command parser evaluation  
P04: ASR and parser integration  
P05: Visual target search  
P06: Strict visual match with distractors  
P07: Inverse kinematics reachability  
P08: Virtual grasp and lift  
P09: Robot-environment contact monitoring  
P10: Handover and basket placement  
P11: Manipulation pipeline integration  
P12: Formal end-to-end manipulation evaluation from valid text commands

## Important Scope

P11 and P12 use valid text commands as input and do not perform ASR inference
during the same trial.

Object grasping is implemented as a virtual grasp using a PyBullet fixed
constraint. Therefore, the results do not represent physical gripper force or
force-closure validation.

Wheelchair movement between predefined stations is simulated when required.
Autonomous navigation, mapping, and global path planning are outside the scope
of this study.

## Repository Structure

```text
src/        Main system implementation
tests/      Evaluation scripts P01 to P12
config/     System configuration files
docs/       Supporting documentation
