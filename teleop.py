#!/usr/bin/env python3
import curses, RPi.GPIO as GPIO, time

# Pin map
AIN1, AIN2 = 5, 6
BIN1, BIN2 = 13, 19
PWM_FREQ = 1000

GPIO.setmode(GPIO.BCM)
for p in (AIN1, AIN2, BIN1, BIN2):
    GPIO.setup(p, GPIO.OUT)
pwm = {
    'a_fwd': GPIO.PWM(AIN1, PWM_FREQ),
    'a_rev': GPIO.PWM(AIN2, PWM_FREQ),
    'b_fwd': GPIO.PWM(BIN1, PWM_FREQ),
    'b_rev': GPIO.PWM(BIN2, PWM_FREQ),
}
for ch in pwm.values():
    ch.start(0)

def set_motor(a, dir, duty):
    # a=True for motor A, False for B; dir=1 fwd, -1 rev, 0 brake
    if a:
        f, r = pwm['a_fwd'], pwm['a_rev']
    else:
        f, r = pwm['b_fwd'], pwm['b_rev']
    if dir > 0:
        f.ChangeDutyCycle(duty); r.ChangeDutyCycle(0)
    elif dir < 0:
        f.ChangeDutyCycle(0); r.ChangeDutyCycle(duty)
    else:
        f.ChangeDutyCycle(0); r.ChangeDutyCycle(0)

def drive(left, right, speed):
    duty = max(0, min(100, speed))
    set_motor(True,  1 if left  > 0 else (-1 if left  < 0 else 0), duty if left  != 0 else 0)
    set_motor(False, 1 if right > 0 else (-1 if right < 0 else 0), duty if right != 0 else 0)

def stop_all():
    for ch in pwm.values(): ch.ChangeDutyCycle(0)

def main(stdscr):
    stdscr.nodelay(True)
    stdscr.keypad(True)
    speed = 50
    left = right = 0
    stdscr.addstr(0,0,"Arrows=drive, A/D=spin, X=stop, +/- speed, Q=quit")
    while True:
        c = stdscr.getch()
        if c == -1:
            time.sleep(0.01)
            continue
        if c in (ord('q'), ord('Q')):
            break
        elif c == curses.KEY_UP:       left, right = 1, 1
        elif c == curses.KEY_DOWN:     left, right = -1, -1
        elif c == curses.KEY_LEFT:     left, right = -1, 1
        elif c == curses.KEY_RIGHT:    left, right = 1, -1
        elif c in (ord('a'), ord('A')): left, right = -1, 1   # in-place left spin
        elif c in (ord('d'), ord('D')): left, right = 1, -1   # in-place right spin
        elif c in (ord('x'), ord('X')): left, right = 0, 0
        elif c == ord('+'):            speed = min(100, speed + 5)
        elif c == ord('-'):            speed = max(10, speed - 5)
        drive(left, right, speed)
        stdscr.addstr(1,0,f"Left={left:+d} Right={right:+d} Speed={speed:3d}%   ")
    stop_all()

try:
    curses.wrapper(main)
finally:
    stop_all()
    GPIO.cleanup()
