import parser as file_parser

# def main():
parser = file_parser.get_parser()

args = parser.parse_args()

print(args)

# if __name__ == "__main__":
#     main()