import os

def assert_directory(directory):
    if directory is not None:
        if not os.path.exists(directory):
            os.mkdir(directory)
        elif len(os.listdir(directory)) != 0:
            acceptable = False
            while(not acceptable):
                overwrite = input("The directory:{} is not empty, meaning that it is possible to overwrite the existing data.\
                    Are you sure to continue? Press \'y\' for yes, and \'n\' for no: ".format(directory))
                if (overwrite != 'y' and overwrite != 'n'):
                    print("Invalid input, try again")
                else:
                    acceptable=True
            if not (overwrite == 'y'):
                print("Please change the saving directory setting")
                exit()