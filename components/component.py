import curses
import logging

logger = logging.getLogger()


class Component:
    name = 'Base'

    def __init__(self, components=None, verbose_flag=True, log_flag=False):
        self.verbose_flag = verbose_flag
        self.log_flag = log_flag
        self.components = {}
        if components:
            self.components.update(components)
        self.screen = None
        self.log('init', force=True)

    def run(self, *args, **kwargs):
        self.log('run: begin')
        result = self._run(*args, **kwargs)
        self.log('run: end')
        return result

    def _run(self, *args, **kwargs):
        raise NotImplementedError

    def log(self, message=None, force=False):
        if not self.log_flag and not force:
            return

        message = f'{self.name}: {message}'
        logger.info(message)
        self.verbose(message, new_line=True)

    def verbose(self, message=None, new_line=False):
        if not self.verbose_flag:
            return

        if new_line:
            self._clean_screen()
            print(message)
        else:
            if not self.screen:
                self.screen = curses.initscr()
                curses.noecho()
                curses.cbreak()
            self.screen.addstr(0, 0, message + '\n')
            self.screen.refresh()

    def __del__(self):
        self._clean_screen()

    def _clean_screen(self):
        if self.screen is not None:
            self.screen = None
            curses.echo()
            curses.nocbreak()
            curses.endwin()
