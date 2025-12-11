
import math
import sys
import pygame


WIDTH, HEIGHT = 900, 900
FPS = 120
BG_COLOR = (18, 18, 20)
ARM_COLOR = (220, 220, 230)
TARGET_COLOR = (255, 120, 120)
JOINT_COLOR = (120, 200, 255)
TEXT_COLOR = (210, 210, 220)

L1 = 220
L2 = 180

BASE_X, BASE_Y = WIDTH // 2, HEIGHT // 2 

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def solve_2link_ik(px, py, elbow_up=True):
    """
    Closed-form IK for a planar 2-link arm in 2D.
    Inputs:
        px, py: target in SCREEN coords (y down). We'll convert to math coords (y up).
        elbow_up: bool -> choose the "elbow up" (+acos) or "elbow down" (-acos) branch.
    Returns:
        theta1, theta2 (radians), elbow_pos, wrist_pos, reached_target (bool), clamped_target_screen (x,y)
    """
    tx = px - BASE_X
    ty = (BASE_Y - py)

    # Distance to target
    r = math.hypot(tx, ty)
    max_reach = L1 + L2
    min_reach = abs(L1 - L2)

    clamped = False

    if r > max_reach:
        if r > 1e-9:
            scale = max_reach / r
        else:
            scale = 0.0
        tx *= scale
        ty *= scale
        r = max_reach
        clamped = True

    if r < min_reach:
        scale = (min_reach / r) if r > 1e-9 else 0.0
        tx *= scale
        ty *= scale
        r = min_reach
        clamped = True

    cos2 = clamp((r*r - L1*L1 - L2*L2) / (2.0 * L1 * L2), -1.0, 1.0)
    base_theta2 = math.acos(cos2)
    theta2 = (+base_theta2) if elbow_up else (-base_theta2)

    phi = math.atan2(ty, tx)
    k1 = L1 + L2 * math.cos(theta2)
    k2 = L2 * math.sin(theta2)
    theta1 = phi - math.atan2(k2, k1)

    ex = L1 * math.cos(theta1)
    ey = L1 * math.sin(theta1)
    wx = ex + L2 * math.cos(theta1 + theta2)
    wy = ey + L2 * math.sin(theta1 + theta2)

    elbow_screen = (int(BASE_X + ex), int(BASE_Y - ey))
    wrist_screen = (int(BASE_X + wx), int(BASE_Y - wy))

    clamped_target_screen = (int(BASE_X + tx), int(BASE_Y - ty))

    return theta1, theta2, elbow_screen, wrist_screen, (not clamped), clamped_target_screen

def angle_deg(rad):
    return (rad * 180.0 / math.pi)

def main():
    pygame.init()
    pygame.display.set_caption("Two-Joint Closed-Form IK (Pygame)")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)
    big_font = pygame.font.SysFont(None, 28)

    elbow_up = True
    freeze_target = False
    target_pos = (WIDTH // 2, HEIGHT // 2 - 40)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    elbow_up = not elbow_up
                elif event.key == pygame.K_f:
                    freeze_target = not freeze_target
                elif event.key == pygame.K_r:
                    elbow_up = True
                    freeze_target = False
                    target_pos = (WIDTH // 2, HEIGHT // 2 - 40)

        if not freeze_target:
            mx, my = pygame.mouse.get_pos()
            target_pos = (mx, my)

        theta1, theta2, elbow_xy, wrist_xy, reached, clamped_target = solve_2link_ik(
            target_pos[0], target_pos[1], elbow_up=elbow_up
        )

        screen.fill(BG_COLOR)

        max_reach = L1 + L2
        min_reach = abs(L1 - L2)
        try:
            pygame.draw.circle(screen, (45, 45, 55), (BASE_X, BASE_Y), max_reach, 1)
            if min_reach > 1:
                pygame.draw.circle(screen, (45, 45, 55), (BASE_X, BASE_Y), min_reach, 1)
        except Exception:
            pass

        pygame.draw.circle(screen, TARGET_COLOR, target_pos, 6)
        if not reached:
            pygame.draw.circle(screen, (255, 200, 80), clamped_target, 4, 1)
            pygame.draw.line(screen, (120, 120, 120), target_pos, clamped_target, 1)

        pygame.draw.line(screen, ARM_COLOR, (BASE_X, BASE_Y), elbow_xy, 6)
        pygame.draw.line(screen, ARM_COLOR, elbow_xy, wrist_xy, 6)

        pygame.draw.circle(screen, JOINT_COLOR, (BASE_X, BASE_Y), 10)
        pygame.draw.circle(screen, JOINT_COLOR, elbow_xy, 9)
        pygame.draw.circle(screen, JOINT_COLOR, wrist_xy, 8)

        lines = [
            "Two-Joint Closed-Form IK",
            f"[SPACE] Toggle Elbow Mode: {'UP' if elbow_up else 'DOWN'}",
            "[F] Freeze/Unfreeze Target  |  [R] Reset  |  [ESC] Quit",
            f"θ1: {angle_deg(theta1):6.2f}°   θ2: {angle_deg(theta2):6.2f}°",
            f"Target {'reached' if reached else 'clamped to reach boundary'}",
            f"Base: ({BASE_X}, {BASE_Y})  L1={L1}  L2={L2}"
        ]

        y = 10
        for i, text in enumerate(lines):
            surf = (big_font if i == 0 else font).render(text, True, TEXT_COLOR)
            screen.blit(surf, (12, y))
            y += (32 if i == 0 else 20)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
