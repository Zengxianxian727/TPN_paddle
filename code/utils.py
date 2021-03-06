import os
import signal

__all__ = ['AttrDict']


def _term(sig_num, addition):
    print('current pid is %s, group id is %s' % (os.getpid(), os.getpgrp()))
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


signal.signal(signal.SIGTERM, _term)
signal.signal(signal.SIGINT, _term)


class AttrDict(dict):

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = AttrDict(value)
        return value


        

    def __setattr__(self, key, value):
        #print('come in setattr')
        if isinstance(value, dict):
            value = AttrDict(value)
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value
            #self.__dict__[key] = value

