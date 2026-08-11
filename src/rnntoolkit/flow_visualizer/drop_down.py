import pygame

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (220, 220, 220)
MEDIUM_GRAY = (150, 150, 150)
DARK_GRAY = (80, 80, 80)
CANVAS_BG = (245, 245, 250)


class DropdownMenu:
    """A vertical menu anchored below a button.

    Clicking an item returns its label; clicking outside closes the menu.
    """

    def __init__(self, file_button, items):
        # The anchor button determines where the menu is positioned.
        self.file = file_button
        self.items = items
        self.visible = False
        self.font = pygame.font.SysFont("Arial", 14)
        self.width = 180
        self.item_height = 29
        self.padding = 5
        self.hovered_index = None

    def _menu_rect(self):
        return pygame.Rect(
            self.file.rect.left,
            self.file.rect.bottom + 4,
            self.width,
            len(self.items) * self.item_height + self.padding * 2,
        )

    def _item_rect(self, index):
        menu_rect = self._menu_rect()
        return pygame.Rect(
            menu_rect.left + self.padding,
            menu_rect.top + self.padding + index * self.item_height,
            menu_rect.width - self.padding * 2,
            self.item_height,
        )

    def show(self):
        """Open the menu."""
        self.visible = True

    def hide(self):
        """Close the menu."""
        self.visible = False

    def handle_event(self, event):
        """Return the selected item's label, or None if none was selected."""
        if not self.visible:
            return None

        if event.type == pygame.MOUSEMOTION:
            self.hovered_index = None
            for i in range(len(self.items)):
                if self._item_rect(i).collidepoint(event.pos):
                    self.hovered_index = i
                    break

        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, item in enumerate(self.items):
                if self._item_rect(i).collidepoint(event.pos):
                    self.hide()
                    return item
            if not self._menu_rect().collidepoint(event.pos):
                self.hide()
        return None

    def draw(self, screen):
        """Draw the menu background and item labels when it is visible."""
        if not self.visible:
            return

        menu_rect = self._menu_rect()
        shadow_rect = menu_rect.move(0, 3)
        pygame.draw.rect(screen, (205, 205, 210), shadow_rect, border_radius=8)
        pygame.draw.rect(screen, WHITE, menu_rect, border_radius=8)
        pygame.draw.rect(screen, (218, 218, 223), menu_rect, 1, border_radius=8)

        for i, item in enumerate(self.items):
            item_rect = self._item_rect(i)
            hovered = i == self.hovered_index
            if hovered:
                pygame.draw.rect(screen, (45, 116, 246), item_rect, border_radius=5)
            text = self.font.render(item, True, WHITE if hovered else (35, 35, 38))
            screen.blit(
                text,
                (item_rect.left + 10, item_rect.centery - text.get_height() // 2),
            )
