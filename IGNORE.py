import pygame
import numpy as np
import math

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(" Pseudoinverse + Null-Space ")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)

link_lengths = [150, 100, 75]
n_joints = len(link_lengths)
base = np.array([WIDTH//2, HEIGHT//2], dtype=float)

joint_angles = np.radians([45, -30, 15]).astype(float)

max_reach = float(sum(link_lengths))

alpha = 0.2           # task-space step size
lambda_damp = 0.08    # damping for damped least squares pseudoinverse
k_null = 0.4          # null-space ascent gain (manipulability)
use_null = True       # toggle for null-space optimization
finite_h = 1e-3       # finite-difference step for gradient

freeze_target = False
frozen_pos = None

def forward_kinematics(angles):
    points = [tuple(base)]
    x, y = base
    theta = 0.0
    for i in range(n_joints):
        theta += angles[i]
        x += link_lengths[i] * math.cos(theta)
        y += link_lengths[i] * math.sin(theta)
        points.append((x, y))
    return points

def jacobian(angles):
    J = np.zeros((2, n_joints), dtype=float)

    for j in range(n_joints):
        dx = 0.0
        dy = 0.0
        for k in range(j, n_joints):
            theta_k = float(np.sum(angles[:k+1]))
            dx -= link_lengths[k] * math.sin(theta_k)
            dy += link_lengths[k] * math.cos(theta_k)
        J[0, j] = dx
        J[1, j] = dy
    return J

def damped_pseudoinverse(J, lam):
    JJt = J @ J.T
    I2 = np.eye(2)
    return J.T @ np.linalg.inv(JJt + (lam ** 2) * I2)

def radial_unit_from_base(point):
    v = point - base
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([1.0, 0.0]) 
    return v / n

def force_objective(angles):
    ee = np.array(forward_kinematics(angles)[-1])
    u = radial_unit_from_base(ee)
    J = jacobian(angles)
    JJt = J @ J.T
    JJt_reg = JJt + (lambda_damp ** 2) * np.eye(2)
    invJJt = np.linalg.inv(JJt_reg)
    val = - float(u.T @ invJJt @ u) 
    return val

def force_objective_grad(angles, h=finite_h):
    base_val = force_objective(angles)
    grad = np.zeros_like(angles)
    for i in range(n_joints):
        pert = angles.copy()
        pert[i] += h
        grad[i] = (force_objective(pert) - base_val) / h
    return grad

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                joint_angles = np.radians([45, -30, 15]).astype(float)
            elif event.key == pygame.K_SPACE:
                freeze_target = not freeze_target
                frozen_pos = None if not freeze_target else np.array(pygame.mouse.get_pos(), dtype=float)
            elif event.key == pygame.K_f:
                joint_angles[1] *= -1.0
            elif event.key == pygame.K_n:
                use_null = not use_null
            elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                k_null *= 1.25
            elif event.key == pygame.K_MINUS or event.key == pygame.K_UNDERSCORE:
                k_null /= 1.25
            elif event.key == pygame.K_LEFTBRACKET:
                alpha /= 1.25
            elif event.key == pygame.K_RIGHTBRACKET:
                alpha *= 1.25

    screen.fill((18, 18, 20))

    if freeze_target and frozen_pos is not None:
        raw_target = frozen_pos.copy()
    else:
        raw_target = np.array(pygame.mouse.get_pos(), dtype=float)

    vec = raw_target - base
    dist = float(np.linalg.norm(vec))
    if dist > max_reach:
        target = base + vec / dist * max_reach
        clamped = True
    else:
        target = raw_target
        clamped = False

    pts = forward_kinematics(joint_angles)
    ee = np.array(pts[-1])

    e = target - ee
    if np.linalg.norm(e) > 0.5:
        J = jacobian(joint_angles)
        J_pinv = damped_pseudoinverse(J, lambda_damp)
        dtheta_task = J_pinv @ e * alpha

        if use_null:
            N = np.eye(n_joints) - J_pinv @ J
            grad = force_objective_grad(joint_angles)
            gnorm = float(np.linalg.norm(grad))
            if gnorm > 1e-9:
                grad = grad / gnorm
            dtheta_null = N @ (k_null * grad)
        else:
            dtheta_null = 0.0

        dtheta = dtheta_task + dtheta_null
        joint_angles += dtheta

    pygame.draw.circle(screen, (45, 45, 55), base.astype(int), int(max_reach), 1)

    for i in range(len(pts) - 1):
        pygame.draw.line(screen, (220, 220, 230), pts[i], pts[i+1], 6)
        pygame.draw.circle(screen, (120, 200, 255), (int(pts[i][0]), int(pts[i][1])), 8)
    pygame.draw.circle(screen, (120, 200, 255), (int(pts[-1][0]), int(pts[-1][1])), 8)

    pygame.draw.circle(screen, (255, 120, 120) if clamped else (100, 150, 255), target.astype(int), 6)
    if clamped:
        pygame.draw.circle(screen, (255, 200, 80), raw_target.astype(int), 4, 1)
        pygame.draw.line(screen, (120, 120, 120), raw_target.astype(int), target.astype(int), 1)

    deg_angles = [((math.degrees(a) + 540) % 360) - 180 for a in joint_angles]
    w_val = force_objective(joint_angles)
    lines = [
        "3-Link Null-Space IK (maximize outward force)",
        f"Angles (deg): {', '.join(f'{a:6.1f}' for a in deg_angles)}",
        f"|e|: {np.linalg.norm(e):6.2f}   w*: {w_val: .4f}",
        f"alpha: {alpha:.3f}   lambda: {lambda_damp:.3f}   k_null: {k_null:.3f}   null: {'ON' if use_null else 'OFF'}",
        "Controls:",
        "  Mouse = move target   SPACE = freeze/unfreeze   R = reset   ESC = quit",
        "  F = flip posture   N = toggle null-space   +/- = k_null   [ ] = alpha",
    ]
    for i, line in enumerate(lines):
        img = font.render(line, True, (210, 210, 220))
        screen.blit(img, (12, 10 + i * 20))

    pygame.display.flip()
    clock.tick(120)

pygame.quit()
