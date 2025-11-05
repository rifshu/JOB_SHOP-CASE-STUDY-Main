##### \# Optimizing Job Shop Scheduling with Deep Reinforcement Learning

##### 

##### The \*\*Job Shop Scheduling Problem (JSSP)\*\* is a classical NP-hard problem in manufacturing and production planning. This project utilizes \*\*Deep Reinforcement Learning (DRL)\*\*, benchmarked against classical \*\*Heuristic Schedulers\*\* (FIFO and SPT), to find near-optimal scheduling policies in a dynamic, stochastic environment simulated using \*\*SimPy\*\*.

##### 

##### The project aims to significantly \*\*reduce makespan\*\*, \*\*increase machine utilization\*\*, and \*\*decrease job tardiness\*\* compared to traditional methods.

##### 

##### ---

##### 

##### \## 🛠️ Installation and Setup

##### 

##### This project requires Python 3.8+ and several external libraries for simulation and Deep RL training.

##### 

##### \### 1. Clone the Repository

##### If you haven't already cloned the repository, do so using Git:

##### ```bash

##### git clone \[https://github.com/rifshu/JOB\_SHOP-CASE-STUDY-Main](https://github.com/rifshu/JOB\_SHOP-CASE-STUDY-Main)

##### cd JOB\_SHOP-CASE-STUDY-Main



##### \# Install core simulation and RL dependencies

##### pip install simpy numpy gymnasium

##### \# Install Stable Baselines 3 (the DRL framework) with extra tools

##### pip install stable-baselines3\[extra]

