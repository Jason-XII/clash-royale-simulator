# PyGame游戏开发笔记

第一步，用`pygame.init()`初始化内部模块。

第二步，进行如下设置：

```python
self.screen = pygame.display.set_mode((W, H)) # 设置屏幕大小，之后用这个对象在屏幕上作画
self.clock = pygame.time.Clock() # 之后设置fps有用
self.font = pygame.font.Font(None, 18) # 使用默认字体，18px
lbl = self.font.render(e.data.name, True, BLACK) # 中间的参数为True会让字体更光滑
self.screen.blit(lbl, lbl.get_rect(center=(sx, sy+r+10)))
```

