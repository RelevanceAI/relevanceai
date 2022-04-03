import os

from references.refgen import ReferenceGenerator

WALK_DIR = "relevanceai"


def main():
    print("walk_dir = " + WALK_DIR)

    # If your current working directory may change during script execution, it's recommended to
    # immediately convert program arguments to an absolute path. Then the variable root below will
    # be an absolute path as well. Example:
    # walk_dir = os.path.abspath(walk_dir)
    print("walk_dir (absolute) = " + os.path.abspath(WALK_DIR))

    for root, subdirs, files in os.walk(WALK_DIR):
        print("--\nroot = " + root)
        list_file_path = os.path.join(root, "my-directory-list.txt")

        print("list_file_path = " + list_file_path)
        for subdir in subdirs:
            print("\t- subdirectory " + subdir)

        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)

                print("\t- file %s (full path: %s)" % (filename, file_path))

                with open(file_path, "r") as f:
                    f_content = f.read()

                    rg = ReferenceGenerator()
                    content = rg.gen(f_content)


if __name__ == "__main__":
    main()
