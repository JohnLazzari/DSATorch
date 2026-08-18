import pygame

Color = tuple[int, int, int]

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (220, 220, 220)
MEDIUM_GRAY = (150, 150, 150)
DARK_GRAY = (80, 80, 80)
CANVAS_BG = (245, 245, 250)

pygame.init()


class Button:
    """A reusable clickable Pygame button with hover and pressed states.

    ``handle_event`` returns True when a press begins over the button, allowing
    the application to react to the click.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str,
        style: str = "default",
    ) -> None:
        """Initialize the button rectangle, label, style, and font."""
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.style = style
        self.is_hovered = False
        self.is_pressed = False
        self.is_active = False
        self.font = (
            pygame.font.SysFont("Arial", 15)
            if style == "menu"
            else pygame.font.Font(None, 18)
        )

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Update mouse state and report whether a new click began."""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and self.is_hovered:
            self.is_pressed = True
            return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_pressed = False
        return False

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the button using a color that reflects its current state."""
        if self.style == "menu":
            if self.is_pressed or self.is_active:
                pygame.draw.rect(screen, (218, 218, 223), self.rect, border_radius=5)
            elif self.is_hovered:
                pygame.draw.rect(screen, (235, 235, 240), self.rect, border_radius=5)

            text_surface = self.font.render(self.text, True, (35, 35, 38))
            screen.blit(text_surface, text_surface.get_rect(center=self.rect.center))
            return

        color = (
            DARK_GRAY
            if self.is_pressed
            else (MEDIUM_GRAY if self.is_hovered else LIGHT_GRAY)
        )
        pygame.draw.rect(screen, color, self.rect, border_radius=4)

        text_surface = self.font.render(self.text, True, BLACK)
        screen.blit(text_surface, text_surface.get_rect(center=self.rect.center))
