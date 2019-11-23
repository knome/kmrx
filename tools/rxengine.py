#!/usr/bin/env python3

# 
# compiles non-capturing regular expression into an efficient form
# 

import optparse
import sys

##

def debug( *args ):
    sys.stderr.write( ' '.join( str( arg ) for arg in args ) + '\n' )

##

def main():
    options, rx   = getopts()
    operationTree = parse_into_operation_tree( rx )
    start, stop   = create_and_connect_nodes( operationTree )
    
    if options.debug:
        debug()
        debug( 'start', id( start ) )
        debug( 'stop' , id( stop  ) )
        debug()
        debug ('<<<<<<<<<<<<<<<<' )
    
    transitions = extract_transitions( start )
    
    if options.debug:
        show_connections( transitions )
        debug ('>>>>>>>>>>>>>>>>' )
        debug()
        debug( '<<<<<<<<<<<<<<<<' )
    
    backpropagate_free_transitions( transitions )
    
    # we do this again because we may have cut out unrequired intermediate states entirely
    # reducing the number of states the final shifter will need to manipuate
    # 
    transitions = extract_transitions( start )
    
    if options.debug:
        show_connections( transitions )
        debug( '>>>>>>>>>>>>>>>>' )
        debug()
    
    maxIndex = enumerate_transitions( transitions )
    maxChunk = maxIndex // options.chunkSize
    
    if options.debug:
        debug( transitions )
    
    ctriggers, utriggers = group_transitions_by_trigger( transitions )
    
    if options.debug:
        debug()
        debug( 'ctriggers', ctriggers )
        debug( 'utriggers', utriggers )
    
    cmrCTriggers, cmrUTriggers = recode_state_shift_to_chunk_mask_rotate(
        options.chunkSize ,
        ctriggers         ,
        utriggers         ,
    )
    
    if options.debug:
        debug()
        debug( 'cmrCTriggers', cmrCTriggers )
        debug( 'cmrUTriggers', cmrUTriggers )
    
    # the alternate is having all character triggers jump to default,
    # but instead we spread the universal transitions over the specific ones
    # allowing them to combine and reduce to a minimal number of actions to
    # effect the transition at any given character
    # 
    fcmrCTriggers = add_universal_triggers_to_all_character_triggers( cmrCTriggers, cmrUTriggers )
    
    if options.debug:
        debug()
        debug( 'fcmrCTriggers', fcmrCTriggers )
        debug( 'cmrUTriggers', cmrUTriggers )
    
    # mask-combined (character/universal) triggers
    # 
    mcc, mcu = combine_masks_of_transitions_with_equal_chunk_and_rotate(
        fcmrCTriggers, cmrUTriggers
    )
    
    # TODO ( may collapse cases around alternations, \n|\0-match-ends, case-folding, character classes, etc )
    # combine characters with the same set of transforms
    # remove characters that exactly match universal transforms
    
    if options.debug:
        debug()
        debug( 'mcc', mcc )
        debug( 'mcu', mcu )
    
    thats_one_sweet_switch_statement_you_might_say( maxChunk, mcc, mcu )
    
    return

##

def thats_one_sweet_switch_statement_you_might_say( maxChunk, mcc, mcu ):
    print( 'switch( cc ){' )
    
    for kk, mcrs in sorted( mcc.items() ):
        # ugly, but avoids caring about escaping
        print( '  case \'\\x%s\': // %s ' % ( hex( ord(kk) )[2:].ljust(2,'0'), repr( kk ) ) )
        for ii in range( maxChunk + 1 ):
            print( '    next[ %s ] = 0 ;' % ii )
        for cr, mms in mcrs.items():
            print( '    next[ %s ] |= ( (prev[ %s ] & %s) %s %s ); // %s ' % (
                cr[2] ,
                cr[0] ,
                hex( sum( (1 << mm) for mm in mms ) ) + 'ull' ,
                '>>' if cr[1] < 0  else '<<' ,
                -cr[1] if cr[1] < 0 else cr[1] ,
                ', '.join( '(%s,%s)' % (mm,mm+cr[1]) for mm in mms) ,
            ))
        print( '    break;' )
    
    print( '  default:' )
    for ii in range( maxChunk + 1 ):
        print( '    next[ %s ] = 0 ;' % ii )
    for cr, mms in mcu.items():
        print( '    next[ %s ] |= ( (prev[ %s ] & %s) %s %s ); // %s ' % (
            cr[2] ,
            cr[0] ,
            hex( sum( (1 << mm) for mm in mms ) ) + 'ull',
            '>>' if cr[1] < 0  else '<<' ,
            -cr[1] if cr[1] < 0 else cr[1] ,
            ', '.join( '(%s,%s)' % (mm,mm+cr[1]) for mm in mms) ,
        ))
    print( '    break;' )
    print( '}' )

##

def combine_masks_of_transitions_with_equal_chunk_and_rotate(
    cmrCTriggers,
    cmrUTriggers,
):
    outc = {}
    for kk, transitions in cmrCTriggers.items():
        combos = {}
        for transition in transitions:
            ic, im, ir, cc = transition
            key = (ic, ir, cc)
            if key not in combos:
                combos[ key ] = []
            combos[ key ].append( im )
        outc[ kk ] = combos
    
    combos = {}
    for transition in cmrUTriggers:
        ic, im, ir, cc = transition
        key = (ic, ir, cc)
        if key not in combos:
            combos[ key ] = []
        combos[ key ].append( im )
    outu = combos
    
    return outc, outu

##

def add_universal_triggers_to_all_character_triggers( cmrCTriggers, cmrUTriggers ):
    out = {}
    for kk, vv in list( cmrCTriggers.items() ):
        out[ kk ] = vv + cmrUTriggers
    
    return out

##

def recode_state_shift_to_chunk_mask_rotate( chunkSize, ctriggers, utriggers ):
    
    # (initialState, consequentialState)
    # (initialChunk, initialChunkMask, initialChunkRotate, consequentialChunk)
    
    cmrCTriggers = {}
    cmrUTriggers = []
    
    for kk, shifts in ctriggers.items():
        cmrCTriggers[ kk ] = []
        for shift in shifts:
            initial, consequential = shift
            cmrCTriggers[ kk ].append( chunk_mask_rotate( chunkSize, initial, consequential ) )
    
    for shift in utriggers:
        initial, consequential = shift
        cmrUTriggers.append( chunk_mask_rotate( chunkSize, initial, consequential ) )
    
    return cmrCTriggers, cmrUTriggers

##

def chunk_mask_rotate( chunkSize, initial, consequential ):
    initialChunk  = initial // chunkSize
    initialOffset = initial % chunkSize
    
    consequentialChunk  = consequential // chunkSize
    consequentialOffset = consequential % chunkSize
    
    rotation = consequentialOffset - initialOffset
    
    return (initialChunk,initialOffset,rotation,consequentialChunk)

##    

def group_transitions_by_trigger( nodes ):
    grouped = {}
    always  = []
    
    for node in nodes:
        for out in node.outs():
            for cc in out[0].matches():
                if cc == None:
                    always.append( (node.index(),out[1].index() ) )
                else:
                    if cc not in grouped:
                        grouped[ cc ] = []
                    grouped[ cc ].append( (node.index(),out[1].index()) )
    
    return grouped, always
##

def show_connections( nodes ):
    for node in nodes:
        print( id( node ) )
        for out in node.outs():
            print( '...', repr( out[0] ), id( out[1] ) )

##

def backpropagate_free_transitions( nodes ):
    for node in nodes:
        node.backpropagate_free_transitions()
    return
  
##

def enumerate_transitions( transitions ):
    for no, node in enumerate( transitions ):
        node.set_index( no )
    
    return no

##

def extract_transitions( start ):
    pending = [ start ]
    seen    = [ start ]
    while pending:
        current = pending.pop(0)
        for connection in current.connections():
            if connection in seen:
                continue
            else:
                pending.append( connection )
                seen.append( connection )
    return seen

##

def create_and_connect_nodes( operationTree ):
    start = Node()
    stop  = Node()
    end   = Node()
    
    stop.connect( MatchDone(), end )
    
    operationTree.create_and_thread_nodes( start, stop )
    
    return start, end

class Node():
    def __init__( self ):
        self._outs  = []
        self._index = None
        return
    
    def __repr__( self ):
        return '<Node %s %s>' % (
            repr( self._index ) ,
            repr( len( self._outs ) ),
        )
    
    def set_index( self, index ):
        if self._index != None: raise Exception( 'wat' )
        self._index = index
    
    def index( self ):
        if self._index == None: raise Exception( 'nope' )
        return self._index
    
    def connect( self, transition, other ):
        self._outs.append( (transition, other) )
        return
    
    def connections( self ):
        return [ out[1] for out in self._outs ]
    
    def outs( self ):
        return self._outs[:]
    
    def backpropagate_free_transitions( self ):
        remaining = [ out[1] for out in self._outs if out[0] == None ]
        patched   = [ out for out in self._outs if out[0] != None ]
        seen      = remaining[:] + patched[:]
        
        while remaining:
            current = remaining.pop(0)
            outs    = current.outs()
            
            for otherOut in outs:
                if otherOut[0] == None:
                    if otherOut[1] not in seen:
                        remaining.append( otherOut[1] )
                        seen.append( otherOut[1] )
                else:
                    if otherOut not in seen:
                        patched.append( otherOut )
                        seen.append( otherOut )
        
        self._outs = patched
    
## 

def parse_into_operation_tree( rx ):
    
    # (context,contents)
    # 
    stack = [ ('top', []), ('run',[]) ]
    
    for cc in rx:
        
        escaped = False
        
        if stack[-1][0] == 'escape':
            escaped = True
            stack.pop()
            # fallthrough
        
        if (not escaped) and cc == '\\':
            stack.append( ('escape',) )
            continue
        
        if stack[-1][0] == 'range':
            if cc == ']' and len( stack[-1][1] ) != 0:
                mr = MatchRange( stack[-1][1] )
                stack.pop()
                stack[-1][1].append( mr )
                continue
            else:
                stack[-1][1].append( cc )
                continue
        
        if stack[-1][0] == 'repetition':
            raise Exception( 'unimplemented' )
        
        # check for structural characters if we're not escaped
        # 
        if not escaped:
            if cc == '?':
                if not stack[-1][1]: raise Exception( 'dangling ?' )
                ss = stack[-1][1].pop()
                stack[-1][1].append( MatchQuestion( ss ) )
                continue
            
            if cc == '*':
                if not stack[-1][1]: raise Exception( 'danging *' )
                ss = stack[-1][1].pop()
                stack[-1][1].append( MatchStar( ss ) )
                continue
            
            if cc == '+':
                if not stack[-1][1]: raise Exception( 'dangling +' )
                ss = stack[-1][1].pop()
                stack[-1][1].append( MatchPlus( ss ) )
                continue
            
            if cc == '[':
                stack.append( ('range',[]) )
                continue
            
            if cc == ']':
                raise Exception( 'bare ] outside range' )
            
            if cc == '(':
                stack.append( ('alternation',[]) )
                stack.append( ('run',[]) )
                continue
            
            if cc == ')':
                # close current run and alternation
                ss = stack.pop()
                stack[-1][1].append( MatchRun( ss[1] ) )
                if stack[-1][0] == 'alternation':
                    ss = stack.pop()
                    stack[-1][1].append( MatchAlternation( ss[1] ) )
                    continue
                elif stack[-1][0] == 'top':
                    raise Exception( 'bare ) outside group' )
                else:
                    raise Exception( 'wat' )
            
            if cc == '|':
                # close current run within the current alternation
                # stack should be a run here, under an alternation or top
                ss = stack.pop()
                stack[-1][1].append( MatchRun( ss[1] ) )
                stack.append( ('run',[]) )
                continue
            
            if cc == '{':
                raise Exception( 'unimplemented' )
            
            if cc == '}':
                raise Exception( 'unimplemented' )
        
        # otherwise we're just adding a specific matcher
        # the matcher added will differ based on whether the character was escaped
        # unknown escapements will be errors for now, rather than falling through to plain chars
        # 
        
        if escaped:
            
            # unspecial characters
            # 
            if cc in '?*+[](){}|{}.':
                matchType = MatchChar( cc )
            
            # special characters
            # 
            elif cc == 't':
                matchType = MatchChar( '\t' )
            elif cc == 'n':
                matchType = MatchChar( '\n' )
            elif cc == 'r':
                matchType = MatchChar( '\r' )
            elif cc == 'f':
                matchType = MatchChar( '\f' )
            elif cc == 'v':
                matchType = MatchChar( '\v' )
            
            # character classes
            # 
            elif cc == 'd':
                matchType = MatchDigit()
            elif cc == 'D':
                matchType = NotDigit()
            elif cc == 's':
                matchType = MatchWhitespace()
            elif cc == 'S':
                matchType = MatchNotWhitespace()
            elif cc == 'w':
                matchType = MatchWord()
            elif cc == 'W':
                matchType = MatchNotWord()
            
            else:
                raise Exception( 'unknown escape match' )
            
        else: # not escaped
            
            if cc == '.':
                matchType = MatchDot()
            elif cc == '^' and len( stack[-1][1] ) == 0:
                matchType = MatchCaret()
            elif cc == '$':
                matchType = MatchDollar()
            else:
                matchType = MatchChar( cc )
        
        stack[-1][1].append( matchType )
    
    # TODO, check stack is only top and last run to shove into it
    if len( stack ) == 2:
        ss = stack.pop()
        stack[0][1].append( MatchRun( ss[1] ) )
        return MatchAlternation( stack[0][1] )
    else:
        raise Exception( 'didnt close something' )

##

class MatchDone():
    def __init__( self, matchNewlines = True ):
        self._matchNewlines = matchNewlines
        return
    
    def __repr__( self ):
        return '<MatchDone>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )
    
    def matches( self ):
        if self._matchNewlines:
            yield '\n'
        yield '\0'

class MatchChar():
    def __init__( self, cc ):
        self._cc = cc
        return
    
    def __repr__( self ):
        return '<MatchChar %s>' % repr( self._cc )
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )
    
    def matches( self ):
        yield self._cc

class MatchQuestion():
    def __init__( self, mm ):
        self._mm = mm
        return
    
    def __repr__( self ):
        return '<MatchQuestion %s>' % repr( self._mm )
    
    def create_and_thread_nodes( self, start, stop ):
        self._mm.create_and_thread_nodes( start, stop )
        start.connect( None, stop )

class MatchPlus():
    def __init__( self, mm ):
        self._mm = mm
        return
    
    def __repr__( self ):
        return '<MatchPlus %s>' % repr( self._mm )
    
    def create_and_thread_nodes( self, start, stop ):
        loopNode = Node()
        self._mm.create_and_thread_nodes( start, loopNode )
        loopNode.connect( None, stop  )
        loopNode.connect( None, start )
    

class MatchStar():
    def __init__( self, mm ):
        self._mm = mm
        return
    
    def __repr__( self ):
        return '<MatchStar %s>' % repr( self._mm )
    
    def create_and_thread_nodes( self, start, stop ):
        self._mm.create_and_thread_nodes( start, start )
        start.connect( None, stop  )

class MatchDot():
    def __repr__( self ):
        return '<MatchDot>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )
    
    def matches( self ):
        yield None
    
class MatchCaret():
    def __repr__( self ):
        return '<MatchCaret>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )
    
class MatchDollar():
    def __repr__( self ):
        return '<MatchDollar>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )

class MatchRun():
    def __init__( self, run ):
        self._run = run
        return
    
    def __repr__( self ):
        return '<MatchRun %s>' % repr( self._run )
    
    def create_and_thread_nodes( self, start, stop ):
        # this looks a little complex, its to avoid creating unneeded intermediate states
        # that was creating multiple identicle intermediate states 
        # 
        remaining          = self._run[:]
        previous           = start
        danglingTransition = None
        while remaining:
            if danglingTransition != None:
                node = Node()
                danglingTransition.create_and_thread_nodes( previous, node )
                previous = node
            
            danglingTransition = remaining.pop(0)
        
        if danglingTransition != None:
            danglingTransition.create_and_thread_nodes( previous, stop )
        else:
            previous.connect( None, stop )

class MatchAlternation():
    def __init__( self, runs ):
        self._runs = runs
    
    def __repr__( self ):
        return '<MatchAlternation %s>' % repr( self._runs )
    
    def create_and_thread_nodes( self, start, stop ):
        for run in self._runs:
            run.create_and_thread_nodes( start, stop )
        return

class MatchDigit():
    def __repr__( self ):
        return '<MatchDigit>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )

class MatchNotDigit():
    def __repr__( self ):
        return '<MatchNotDigit>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )

class MatchWhitespace():
    def __repr__( self ):
        return '<MatchWhitespace>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )
    
class MatchNotWhitespace():
    def __repr__( self ):
        return '<MatchNotWhitespace>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )

WORD = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'

class MatchWord():
    def __repr__( self ):
        return '<MatchWord>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )
    
    def matches( self ):
        for cc in WORD:
            yield cc

class MatchNotWord():
    def __repr__( self ):
        return '<MatchNotWord>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )
    
    def matches( self ):
        for ii in range( 256 ):
            cc = chr( ii )
            if cc not in WORD:
                yield cc

##

def getopts():
    parser = optparse.OptionParser()
    
    parser.add_option(
        '-d', '--debug',
        action = 'store_true',
        dest   = 'debug',
        help   = 'output debugging information',
    )
    
    parser.add_option(
        '-c', '--chunkSize',
        dest    = 'chunkSize' ,
        type    = 'int' ,
        default = 64 ,
        help    = 'size to use for chunking out state data (8,16,32,64,128)'
    )
    
    parser.add_option(
        '-n', '--name',
        dest    = 'functionName',
        default = 'knomerx',
        help    = 'a prefix for the regex bits',
    )
    
    options, args = parser.parse_args()
    if len( args ) != 1:
        debug( 'rxengine accepts one free argument, the regular expression to compile' )
        sys.exit(1)
    else:
        return options, args[0]

##

if __name__ == '__main__':
    main()
