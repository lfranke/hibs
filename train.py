import hibs
from hibs import env

env.configure()

from hibs.args import get_args
from hibs.task import HibsTask

if __name__ == "__main__":
    HibsTask(get_args()).run()
