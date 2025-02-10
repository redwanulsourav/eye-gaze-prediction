import os
import json
import yaml
import argparse

from database_obj import Database

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p','--prefix')
    ap = ap.parse_args()

    db = Database(ap.prefix)


    while True:
        print('>>> ', end='')
        command = input()
        command = command.split()
        if command[0] == 'find':
            print(db.find(command[1]))
        elif command[0] == 'findmin':
            if len(command) == 3:
                print(db.findmin(command[1], command[2]))
            else:
                print(db.findmin('',command[1]))
        elif command[0] == 'findargmin':
            print(db.findargmin('', command[1]))
        elif command[0] == 'findbestepoch':
            run_id = 
        
