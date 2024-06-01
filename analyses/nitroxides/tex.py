
import pandas

def format_table(data: pandas.DataFrame, titles: list, line_maker, column_def=None) -> str:
        if column_def is None:
            column_def = ['X[c]'] * len(titles)
            column_def[0] = '>{\\bfseries}l'
        
        r = '\\begin{{tblr}}{{{}}}\n\\hline\n'.format(''.join(column_def))
        
        # titles
        r += ' & '.join(titles) + '\\\\\n\\hline\n'
        
        # data
        for row in data.iterrows():
            r += ' & '.join(line_maker(row[1])) + '\\\\\n'
        
        r += '\\hline\n\\end{tblr}'
        
        return r

def format_longtable(data: pandas.DataFrame, titles: list, line_maker, column_def=None) -> str:
        if column_def is None:
            column_def = ['X[c]'] * len(titles)
            column_def[0] = '>{\\bfseries}l'
        
        r = '\\begin{{longtblr}}[caption={{}}]{{colspec={{{}}}, width = 0.85\\linewidth,rowhead=1}}\n\\hline\n'.format(''.join(column_def))
        
        # titles
        r += ' & '.join(titles) + '\\\\\n\\hline\n'
        
        # data
        for row in data.iterrows():
            r += ' & '.join(line_maker(row[1])) + '\\\\\n'
        
        r += '\\hline\n\\end{longtblr}\n'
        
        return r
