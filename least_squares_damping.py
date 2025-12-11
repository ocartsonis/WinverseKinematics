import pygame
import numpy as np
import math

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3-Link IK (Pseudoinverse)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)

link_lengths = [150, 100, 75]
n_joints = len(link_lengths)
joint_angles = np.radians([45, -30, 15]) 

max_reach = sum(link_lengths)

freeze_target = False
frozen_pos = None

def forward_kinematics(angles):
    """Return positions of each joint and end effector."""
    points = [(WIDTH//2, HEIGHT//2)] 
    x, y = points[0]
    theta = 0
    for i in range(n_joints):
        theta += angles[i]
        x += link_lengths[i] * math.cos(theta)
        y += link_lengths[i] * math.sin(theta)
        points.append((x, y))
    return points

def jacobian(angles):
    """Compute Jacobian matrix for end effector position wrt angles."""
    J = np.zeros((2, n_joints))
    end_x, end_y = forward_kinematics(angles)[-1]
    for j in range(n_joints):
        dx, dy = 0, 0
        for k in range(j, n_joints):
            theta = sum(angles[:k+1])
            dx -= link_lengths[k] * math.sin(theta)
            dy += link_lengths[k] * math.cos(theta)
        J[0, j] = dx
        J[1, j] = dy
    return J

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                joint_angles = np.radians([45, -30, 15])
            elif event.key == pygame.K_SPACE:
                freeze_target = not freeze_target
                frozen_pos = None if not freeze_target else np.array(pygame.mouse.get_pos())

    screen.fill((30, 30, 30))

    if freeze_target and frozen_pos is not None:
        raw_target = frozen_pos
    else:
        raw_target = np.array(pygame.mouse.get_pos())

    base = np.array([WIDTH//2, HEIGHT//2])
    offset = raw_target - base
    dist = np.linalg.norm(offset)
    if dist > max_reach:
        target = base + offset / dist * max_reach
        clamped = True
    else:
        target = raw_target
        clamped = False

    points = forward_kinematics(joint_angles)
    end_effector = np.array(points[-1])

    error = target - end_effector
    if np.linalg.norm(error) > 1.0:
        J = jacobian(joint_angles)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        lambda_ = 0.5 * np.max(S)
        JJT = J @ J.T
        dtheta = J.T @ np.linalg.inv(JJT + (lambda_**2) * np.eye(2)) @ error * 0.2
        joint_angles += dtheta

    pygame.draw.circle(screen, (80, 80, 80), base.astype(int), max_reach, 1)

    for i in range(len(points)-1):
        pygame.draw.line(screen, (200,200,200), points[i], points[i+1], 4)
        pygame.draw.circle(screen, (255,100,100), (int(points[i][0]), int(points[i][1])), 6)
    pygame.draw.circle(screen, (100,255,100), (int(points[-1][0]), int(points[-1][1])), 8)

    color = (255,100,100) if clamped else (100,100,255)
    pygame.draw.circle(screen, color, target.astype(int), 6)

    deg_angles = [math.degrees(a) % 360 for a in joint_angles]
    lines = [
        "3-Link Pseudoinverse IK",
        f"Angles: {', '.join(f'{ang:6.1f}' for ang in deg_angles)}",
        "Controls:",
        "  Mouse = move target",
        "  SPACE = freeze/unfreeze target",
        "  R = reset pose",
        "  ESC = quit"
    ]
    for i, line in enumerate(lines):
        img = font.render(line, True, (220,220,220))
        screen.blit(img, (10, 10 + i*20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()