import openmc.data
from misc_library import welcome_message
import sys
import tempfile

empty_data = ' 0.000000+0 0.000000+0          0          0          0          0'
content_width = len(empty_data)
empty_mat = '   0'
empty_mf = ' 0'
empty_mt = '  0'
end_line_number, empty_line_number = '99999', '    0'

""" 
module to read endf files, without missing their decay data.

.asc files differs from .endf files in that they use replace_empty_data_with_space= True for all of the following functions

Turns out line number is an entirely optional thing as all endf record only matters in the first 75 characters of each line.
Past that (from the 76th character onwards of each line) and it's useless (equivalent to comments).
Generally .endf files would use this space as the line number counter for each section.
Perhaps it would be easier to read the data into a numpy matrix, transpose it, and then split it down by column to fit into:
    [data, data, data, data, data, data], mat, mf, mt, lino
And the whole file should have the SAME length for every line. I.e. =75 if w/o lino, =80 if w/ lino.
"""
def TEXT(mat=1, mf=0, mt=0, text=empty_data):
    """
    Write the TEXT record (i.e. line) as the very first line for each tape.
    text, MAT, MF, MT//
    """
    assert len(text)==content_width, f"The header text must have exactly width={content_width}"
    MAT = str(mat).rjust(4)
    MF = str(mf).rjust(2)
    MT = str(mt).rjust(3)
    LINO = str(0).rjust(5)
    return text+MAT+MF+MT+LINO+'\n'

def SEND(mat, mf, *args, replace_empty_data_with_space=False):
    """
    Section END record line.
    ... MAT, MF, MT=0, (, line number=9999)//
    where the '...' represent an empty data string.
    Note that in the ENDF manual, the ^ line above is written in a weird way,
    '[MAT,MF, 0/ 0.0, 0.0, 0, 0, 0, 0] SEND'
    within the square bracket are the numbers that will actuallly be found in that line of the endf file;
        to the left of the '/' are the things on the right side of that line;
        to the right of the '/' are the things to be written on the left side of the line.
        In otherwords to reconstruct the line,
            split up the 'MAT,MF, 0/ 0.0, 0.0, 0, 0, 0, 0' line down the '/',
            and then flip the order of these two halfs, stitch them back together,
            fill in the 'MAT' and 'MF' with the appropriate numbers,
            and add the line number '99999' after it (since it's a non-data line)
        The word 'SEND' is just the name of this line format.
    asc files have replace_empty_data_with_space=True for all of these lines.
    """
    DATA = ' '*content_width if replace_empty_data_with_space else empty_data
    MAT = str(mat).rjust(4)
    MF = str(mf).rjust(2)
    MT = empty_mt
    LINO = end_line_number
    return DATA+MAT+MF+MT+LINO+'\n'

def FEND(mat, *args, replace_empty_data_with_space=False):
    """
    File END record line.
    ... MAT, 0, 0, 0//
    See SEND.__doc__ for more
    """
    DATA = ' '*content_width if replace_empty_data_with_space else empty_data
    MAT = str(mat).rjust(4)
    MF = empty_mf
    MT = empty_mt
    LINO = empty_line_number
    return DATA+MAT+MF+MT+LINO+'\n'

def MEND(*args, replace_empty_data_with_space=False):
    """
    Material END record line.
    ... 0, 0, 0, 0//
    See SEND.__doc__ for more
    """
    DATA = ' '*content_width if replace_empty_data_with_space else empty_data
    MAT = empty_mat
    MF = empty_mf
    MT = empty_mt
    LINO = empty_line_number
    return DATA+MAT+MF+MT+LINO+'\n'

def TEND(*args, replace_empty_data_with_space=True):
    """
    Tape END record line, i.e. end of an entire library.
    ... -1, 0, 0, 0//
    See SEND.__doc__ for more
    """
    DATA = ' '*content_width if replace_empty_data_with_space else empty_data
    MAT = '  -1'
    MF = empty_mf
    MT = empty_mt
    LINO = empty_line_number
    return DATA+MAT+MF+MT+LINO+'\n'

def endf_print(endf_section):
    if isinstance(endf_section, str):
        endf_section = endf_section.split('\n') # concatenate a list of strings back into a single paragraph. Each sentence(element) is considered as a new line.
    else:
        assert isinstance(endf_section, list) and isinstance(endf_section[0], str), "endf_print is used for printing raw endf file texts only."
    title_line = []
    for i in range(6):
        title_line.append( ('data col' + str(i+1)).rjust(11) )
    title_line.append( 'line number(optional)'.rjust(11) )
    print("|".join(title_line))
    print("_"*len(endf_section[0]))

    def endf_line_splitter(line):
        split_line = [ line[i*11:(i+1)*11] for i in range(6) ]+[line[66:70], line[70:72], line[72:75]]
        if len(line)>75:
            split_line.append(line[75:])
        return split_line

    prev_mat, prev_mf, prev_mt = '   1',' 0','  0'
    for line in endf_section:
        split_line = endf_line_splitter(line)
        new_mat, new_mf, new_mt = split_line[6:9]
        if prev_mat!=new_mat:
            print("Material ID =".ljust(22), new_mat)
            prev_mat = new_mat
        if prev_mf!=new_mf:
            print("MF (ENDF file section) =".ljust(22), new_mf)
            prev_mf = new_mf
        if prev_mt!=new_mt:
            print("MT (reaction number) =".ljust(22), new_mt)
            prev_mt = new_mt
        print("|".join(split_line[:6]+split_line[9:])) # ignore the mat, mf, mt.

if __name__=='__main__': # a one-off probe to 
    endf_file_list = welcome_message()
    input('Press enter to start checking the files previously read for convertability into openmc.data.decay.Decay objects')
    decayable_list = [ repr(nuclide) for nuclide in endf_file_list if len([i for i in nuclide.reaction_list if i[:2]==(8, 457)])>0 ]
    print(decayable_list)
    print(len(decayable_list), " materials were found to be decayable.")

if False:
    # quick fix for R&D
    """
    sys.argv = [sys.argv[0]] 
    sys.argv.append('ChipIR/')
    sys.argv.append('ENDF_B/')
    welcome_message()
    """
    irdff_split = openmc.data.Evaluation('IRDFF_split/Ag109_neutron.endf')
    irdff = [i for i in openmc.data.get_evaluations('IRDFF/IRDFFII.endf') if 'Ag-109' in repr(i)][0]
    eaf = openmc.data.Evaluation('EAF/Ag109.endf')
    fendl = openmc.data.Evaluation('FENDL/Ag109.endf')
    tendl = openmc.data.Evaluation('TENDL/gxs-709/Ag109g.asc')
    endf_b_inc = openmc.data.Evaluation('ENDF_B/endfb80-n/gxs-709/Ag109g.asc')
    endf_b_dec = openmc.data.Evaluation('ENDF_B/decay/Ag109')

    neutron_mf = [1, 2, 3]
    decay_mf = [1, 8, 9, 10]
    radionuclide_multiplicity_mf = [1,9]
    radionuclide_production_xs_mf = [1,10]
    accepted_mf = decay_mf

    endf_file = eaf # choose eaf this time

    # with tempfile.TemporaryFile(mode='w+') as tf:
    tf = open('temp.txt','w+')
    tf.write(TEXT()) # write a comment line
    selected_sections = [(key, val) for (key, val) in endf_file.section.items() if key[0] in accepted_mf]
    section_names, section_content = list(zip(*selected_sections))
    for ind, (mf_mt, content) in enumerate(zip(section_names, section_content)):
        #if change mf: add the 000 line
        tf.write(content)
        tf.write(SEND(endf_file.material, mf_mt[0])) # write end-of-section line
        if (ind+1)<len(section_names):
            if section_names[ind][0]!=section_names[ind+1][0]: # if next section is of a different MF:
                tf.write(FEND(endf_file.material)) # Write end-of-file line
        else:  # end of material
            tf.write(MEND())
    tf.write(TEND()) # end of file
    tf.seek(0)
    ev = openmc.data.Decay(tf)
    tf.close()

    # WTF why are some entries working and some not