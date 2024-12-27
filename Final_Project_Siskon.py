import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D

# Spesifikasi robot
casis_length = 0.16   # panjang casis (m)
casis_width = 0.06    # lebar casis (m)
casis_height = 0.09   # tinggi casis (m)
wheel_diameter = 0.065  # diameter roda (m)
wheel_width = 0.027    # lebar roda (m)

total_mass = 0.7  # Berat total (kg)
g = 9.81  # Gravitasi (m/s^2)

# Pendulum parameter (robot miring sebagai pendulum)
initial_angle = np.pi / 6  # sudut awal (30 derajat)
angular_velocity = 0.0  # kecepatan sudut awal (rad/s)
time_step = 0.02  # waktu per langkah simulasi (s)

# Fungsi simulasi pendulum dengan kontrol PID


def simulate_pendulum(theta, omega, dt, control_torque):
    # Persamaan gerak sederhana untuk pendulum (dengan kontrol torsi)
    alpha = -(g / (casis_height / 2)) * np.sin(theta) + control_torque
    omega += alpha * dt
    theta += omega * dt
    return theta, omega

# Fungsi PID controller


def pid_controller(theta, theta_target, integral, previous_error, dt, Kp, Ki, Kd):
    # Hitung error dan komponen PID
    error = theta - theta_target
    integral += error * dt
    derivative = (error - previous_error) / dt

    # Komponen PID
    control_torque = -Kp * error - Ki * integral - Kd * derivative

    # Kembalikan kontrol torsi, integral, dan error sebelumnya
    return control_torque, integral, error


# Cost function ITAE (Integral of Time-weighted Absolute Error)
def cost_function(Kp, Ki, Kd, theta_initial, omega_initial, time_steps=300, time_step=0.02):
    theta = theta_initial
    omega = omega_initial
    integral = 0.0
    previous_error = 0.0
    cost = 0.0
    for t in range(time_steps):
        # Menghitung torsi kontrol menggunakan PID
        control_torque, integral, previous_error = pid_controller(
            theta, 0.0, integral, previous_error, time_step, Kp, Ki, Kd)

        # Simulasi gerakan pendulum
        theta, omega = simulate_pendulum(
            theta, omega, time_step, control_torque)

        # Akumulasi error absolut yang diberi bobot waktu untuk ITAE
        cost += (t * time_step) * abs(theta)  # Bobot waktu untuk ITAE

    return cost




# Implementasi algoritma ABC untuk optimasi PID


def abc_optimization(iterations, colony_size, bounds):
    # Inisialisasi populasi solusi
    population = np.random.uniform(
        bounds[0], bounds[1], (colony_size, 3))  # (Kp, Ki, Kd)
    fitness = np.zeros(colony_size)

    # Evaluasi awal solusi
    for i in range(colony_size):
        Kp, Ki, Kd = population[i]
        fitness[i] = cost_function(Kp, Ki, Kd, initial_angle, angular_velocity)

    # Main loop untuk ABC
    best_solution = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    for iteration in range(iterations):
        # Employee bee phase
        for i in range(colony_size):
            Kp, Ki, Kd = population[i]
            new_solution = population[i] + \
                np.random.uniform(-1, 1, 3) * np.array([Kp, Ki, Kd])
            new_solution = np.clip(new_solution, bounds[0], bounds[1])
            new_fitness = cost_function(
                new_solution[0], new_solution[1], new_solution[2], initial_angle, angular_velocity)

            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        # Onlooker bee phase (memilih solusi terbaik berdasarkan probabilitas)
        total_fitness = np.sum(fitness)
        # Inversely proportional to cost
        probabilities = (1 / (1 + fitness)) / total_fitness
        for i in range(colony_size):
            if np.random.rand() < probabilities[i]:
                Kp, Ki, Kd = population[i]
                new_solution = population[i] + \
                    np.random.uniform(-1, 1, 3) * np.array([Kp, Ki, Kd])
                new_solution = np.clip(new_solution, bounds[0], bounds[1])
                new_fitness = cost_function(
                    new_solution[0], new_solution[1], new_solution[2], initial_angle, angular_velocity)

                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness

        # Scout bee phase (mencari solusi baru secara acak)
        for i in range(colony_size):
            if np.random.rand() < 0.1:  # Toleransi rendah untuk scout bee
                population[i] = np.random.uniform(bounds[0], bounds[1], 3)
                fitness[i] = cost_function(
                    population[i][0], population[i][1], population[i][2], initial_angle, angular_velocity)

        # Update best solution
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        print(
            f"Iteration {iteration+1}/{iterations}, Best Fitness: {best_fitness}")

    return best_solution

# Fungsi untuk menggambar robot


def draw_robot(ax, theta):
    ax.clear()
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.1, 0.3)
    ax.set_aspect('equal')

    # Posisi roda (lingkaran besar di bawah casis)
    wheel_center = (0, wheel_diameter / 2)
    wheel = Circle(wheel_center, wheel_diameter / 2, color="black")
    ax.add_patch(wheel)

    # Posisi casis (persegi panjang biru yang bergerak seperti pendulum)
    casis_bottom_left = (-casis_height / 2, wheel_diameter)
    casis = Rectangle(casis_bottom_left, casis_height,
                      casis_length, color="blue", alpha=0.7)

    # Transformasi rotasi
    transform = Affine2D().rotate_around(0, wheel_diameter, theta) + ax.transData
    casis.set_transform(transform)
    ax.add_patch(casis)

    ax.set_title("2D Robot Pendulum Simulation")


# Parameter untuk optimasi ABC
iterations = 50
colony_size = 30
bounds = [0.0, 20.0]  # Range untuk Kp, Ki, Kd

# Optimasi parameter PID menggunakan ABC
best_pid = abc_optimization(iterations, colony_size, bounds)
Kp_best, Ki_best, Kd_best = best_pid

# Simulasi dengan parameter PID terbaik
print(f"Best PID Parameters: Kp = {Kp_best}, Ki = {Ki_best}, Kd = {Kd_best}")

# Gambar robot dengan kontrol PID terbaik
fig, ax = plt.subplots()
theta = initial_angle
omega = angular_velocity
integral = 0.0
previous_error = 0.0

for _ in range(300):
    control_torque, integral, previous_error = pid_controller(
        theta, 0.0, integral, previous_error, time_step, Kp_best, Ki_best, Kd_best)
    theta, omega = simulate_pendulum(theta, omega, time_step, control_torque)
    draw_robot(ax, theta)
    plt.pause(time_step)

plt.show()
