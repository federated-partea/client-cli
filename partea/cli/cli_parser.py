import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", dest="username", help="enter your username", type=str, required=True)
    parser.add_argument("-token", dest="token", help="enter your project token", type=str, required=True)
    parser.add_argument("-path", dest="path", help="enter your file path", type=str, required=True)
    parser.add_argument("-pw", dest="password", help="enter your password", type=str, default=None, nargs='?')
    parser.add_argument("-server_url", dest="server_url", help="enter your server", type=str, nargs='?',
                        default='https://partea-api.zbh.uni-hamburg.de/client')
    parser.add_argument("-duration_col", dest="duration_col", help="enter your time column", type=str, nargs='?',
                        default='time', required=True)
    parser.add_argument("-event_col", dest="event_col", help="enter your event column", type=str, nargs='?',
                        default='status', required=True)
    parser.add_argument("-category_col", dest="category_col", help="enter your category column", type=str, nargs='?',
                        default=None)
    parser.add_argument("-sep", dest="sep", help="enter your separator", type=str, nargs='?', default=',')

    args = parser.parse_args()

    return args.username, args.token, args.path, args.password, args.server_url, args.duration_col, args.event_col, \
        args.category_col, args.sep
