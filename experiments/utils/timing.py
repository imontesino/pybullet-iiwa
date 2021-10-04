from datetime import datetime

class timer():
    def __init__(self):
        self.begin = datetime.now()
        self.start_time = datetime.now()

    def start(self):
        self.start_time = datetime.now()

    def end(self, name):
        end_time = datetime.now()
        duration = end_time - self.start_time
        print("--- Timing ---")
        print("Step '{}' duration: {}s {}ms".format(name,
                                                    duration.seconds,
                                                    int(duration.microseconds/1000)))
        print()
        self.start_time = datetime.now()