#!/usr/bin/env python
'''
Description     : Tool to run various analyses on data
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2013.09.20
Last Modified   : 2013.02.27
'''

# TODO: Remove the following line when packaging this module
import os,sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.path.pardir))

def subcommand_summarize(args):
    '''
    Generate a summary report about a data file
    '''
    from datsci import dataio
    
    raise NotImplementedError('Trying to figure out what to do with this binary, since I\'ve been using this project from within ipython notebook')
#     df = load_data(args.file, args.k, args.a)
# 
#     TODO:
#     
# 
#     
#     read in certain columns as categorical text rather than numeric values
#     dtype keyword param read_csv
# 
#     test read random lines from file
# 
#     test read various data formats with different delimiter types
# 

def main():
    import argparse
    import sys
    
    # Create a generic parser with input and output file arguments
    parser_general = argparse.ArgumentParser(add_help=False)
    parser_general.add_argument('file',
                                help='Input data file',
                                nargs='?',
                                type=argparse.FileType('rU'),
                                default=sys.stdin)
    parser_general.add_argument('-f', '--filetype', dest='ft',
                                help='Input file format. Default tsv',
                                choices=['tsv','csv','excel.csv'],
                                default='tsv')
    parser_general.add_argument('-n', '--no-header',
                                help="Set flag to indicate file does not contain a header row",
                                action='store_true')
    parser_general.add_argument('-o', '--out',
                                help='Output file',
                                type=argparse.FileType('wb'),
                                default=sys.stdout)

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Tool to run various analyses on data')
    subparsers = parser.add_subparsers(title='subcommands',
                                       description='Available tools',
                                       dest='subcommand')
    
    # Data summary parser
    parser_summarize = subparsers.add_parser('summarize',
                                             help='Generate summary report about a data file',
                                             parents=[parser_general])
    parser_summarize.add_argument('-k', '--sample-size', dest='k',
                                  help='Number of sub-samples to take from the file',
                                  type=int)
    parser_summarize.add_argument('-r', '--random-seed', dest='rseed',
                                  help='Set random seed for sub-sampling for repeatability',
                                  type=int)
    parser_summarize.set_defaults(func=subcommand_summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

