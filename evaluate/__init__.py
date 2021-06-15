import re
from multiprocessing import Queue, Process
from os import environ
from queue import Empty
from importlib import import_module
from io import StringIO
from contextlib import redirect_stdout
from typing import Tuple

AVAILABLE_MODULES = ['math', 'itertools']
REPLACEMENT_HEADER_PATTERN = re.compile('##@@@@ CODE-REPLACEMENT: ([A-Za-z0-9_]+) by ([A-Za-z0-9_]+) ##')
CODE_REPLACEMENT_PATTERN = '##@@@@ CODE-REPLACEMENT: {key} [^@]* ## CODE-REPLACEMENT END for {key} @@@@##'


def _execute_code(recv: Queue, send: Queue):
    """
    코드를 실행합니다.

    :param Queue send: Queue for sending the result.
        This function will generate Dict[sympy.Expr, sympy.Expr] for the result of solving given expression.
    :param Queue recv: Queue for receiving the expressions to be computed
    """
    _globals = {key: import_module(key)
                for key in AVAILABLE_MODULES}

    _global_with_sympy = _globals.copy()
    _global_with_sympy['sympy'] = import_module('sympy')
    _global_with_sympy['re'] = import_module('re')

    while True:
        try:
            # Receive an object
            code = recv.get(block=True, timeout=600)
            # Wait 600 seconds for messages
        except Empty:
            continue
        except Exception as e:
            send.put(('', '', e))
            continue

        if not code:
            # Break the loop if received_object is false.
            break

        try:
            print('HERE!!!!', code)
            if '##@@@@' in code:
                # Evaluate the code with sympy first
                _locals = {}
                exec(code, _global_with_sympy, _locals)

                while True:
                    matched = REPLACEMENT_HEADER_PATTERN.match(code)
                    if matched is None:
                        break

                    result_key = matched.group(1)
                    result_code = _locals[matched.group(2)]
                    code = re.sub(CODE_REPLACEMENT_PATTERN.format(key=result_key), result_code, code)

            # Evaluate the code
            _stdout = StringIO()
            with redirect_stdout(_stdout):
                exec(code, _globals, {})

            answer = _stdout.getvalue().strip()
            send.put((answer, code, None))
        except Exception as e:
            send.put(('', '', e))

    send.close()
    recv.close()


class Executor(object):
    """
    Class for answer checking purposes.
    """

    def __init__(self, time_limit: float = 0.5):
        """
        Class for evaluating python code

        :param float time_limit:
            maximum amount of allowed time for computation in seconds (default 0.5)
        """

        self.time_limit = time_limit

        self.solver_process = None
        self.to_solver = None
        self.from_solver = None

        if environ.get('DEBUG', False):
            from logging import Logger, INFO
            self._debug_logger = Logger('CodeExec', INFO)
            self._debug = True
        else:
            self._debug = False

        self._start_process()

    def _start_process(self):
        """
        Begin child process for running sympy
        """
        try:
            recv = Queue(maxsize=4)
            send = Queue(maxsize=4)
            self.solver_process = Process(target=_execute_code, name='CodeExec', args=(send, recv))
            self.to_solver = send
            self.from_solver = recv
            self.solver_process.start()

            if self._debug:
                self._debug_logger.warning('CodeExec started with pid = %s', self.solver_process.pid)
        except Exception as e:
            if self._debug:
                self._debug_logger.warning('Exception occurred when starting a child process', exc_info=e)

    def close(self):
        """
        Terminate child process for sympy
        """
        try:
            self.to_solver.put(False)
            self.to_solver.close()
            self.from_solver.close()

            if self.solver_process.is_alive():
                self.solver_process.kill()
            if self._debug:
                self._debug_logger.warning('CodeExec closed with pid = %s', self.solver_process.pid)
        except Exception as e:
            if self._debug:
                self._debug_logger.warning('Exception occurred when closing the child process', exc_info=e)

    def _restart_process(self):
        """
        Restart child process for sympy
        """
        self.close()
        self._start_process()

    def run(self, code: str) -> Tuple[str, str]:
        """
        Evaluate current python code

        :param str code:
            String of python code to evaluate
        :rtype: (str, str)
        :return:
            A python code, and
            Result of executed python code
        """
        solution = []
        try:
            self.to_solver.put(code)
            solution, code, exception = self.from_solver.get(timeout=self.time_limit)
        except Exception as e:
            exception = e
            self._restart_process()
            pass

        if exception:
            if self._debug:
                self._debug_logger.warning('Exception occurred on code execution', exc_info=exception)
            return code, ''
        else:
            return code, solution


__all__ = ['Executor']
