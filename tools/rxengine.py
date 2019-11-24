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
    
    operationTree = parse_into_operation_tree( rx, ignoreCase = options.ignoreCase )
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
    
    if options.debug:
        debug()
        debug( 'mcc', mcc )
        debug( 'mcu', mcu )
    
    cmcc = coalesce_grouped_characters_with_equal_transition_sets( mcc )
    
    if options.debug:
        debug()
        debug( 'cmcc', cmcc )
        debug( 'mcu', mcu )
    
    # TODO ( may collapse cases around alternations, \n|\0-match-ends, case-folding, character classes, etc )
    # remove characters that exactly match universal transforms
    
    switchStatement = thats_one_sweet_switch_statement_you_might_say( maxChunk, cmcc, mcu )
    
    print( switchStatement )
    
    return

##

def thats_one_sweet_switch_statement_you_might_say( maxChunk, cmcc, mcu ):
    bits = []
    
    bits.append( 'switch( cc ){\n' )
    
    for mcrs, kks in cmcc.items():
        bits.append( '  //' )
        cc = 0
        for kk in sorted( kks ):
            if cc == 10:
                cc = 0
                bits.append( '\n  // ' )
            else:
                bits.append( ' ' )
            cc += 1
            bits.append( repr( kk ) )
        bits.append( '\n' )
        
        bits.append( ' ' )
        cc = 0
        for kk in sorted( kks ):
            if cc == 10:
                cc = 0
                bits.append( '\n  ' )
            else:
                bits.append( ' ' )
            cc += 1
            bits.append( 'case \'\\x%s\':' % hex( ord(kk) )[2:].rjust(2,'0') )
        bits.append( '\n' )
        
        for ii in range( maxChunk + 1 ):
            bits.append( '    next[ %s ] = 0 ;\n' % ii )
        
        for cr, mms in mcrs:
            bits.append( '    next[ %s ] |= ( (prev[ %s ] & %s) %s %s ); // %s\n' % (
                cr[2] ,
                cr[0] ,
                hex( sum( (1 << mm) for mm in mms ) ) + 'ull' ,
                '>>' if cr[1] < 0  else '<<' ,
                -cr[1] if cr[1] < 0 else cr[1] ,
                ', '.join( '(%s,%s)' % (mm,mm+cr[1]) for mm in mms) ,
            ))
        
        bits.append( '    break;\n\n' )
    
    bits.append( '  default:\n' )
    for ii in range( maxChunk + 1 ):
        bits.append( '    next[ %s ] = 0 ;\n' % ii )
    for cr, mms in mcu.items():
        bits.append( '    next[ %s ] |= ( (prev[ %s ] & %s) %s %s ); // %s\n' % (
            cr[2] ,
            cr[0] ,
            hex( sum( (1 << mm) for mm in mms ) ) + 'ull',
            '>>' if cr[1] < 0  else '<<' ,
            -cr[1] if cr[1] < 0 else cr[1] ,
            ', '.join( '(%s,%s)' % (mm,mm+cr[1]) for mm in mms) ,
        ))
    bits.append( '    break;\n' )
    bits.append( '}\n' )
    
    return ''.join( bits )

##

def coalesce_grouped_characters_with_equal_transition_sets(
    mcc ,
):
    cmcc = {}
    for kk, mcrs in mcc.items():
        key = tuple( sorted( (cr, tuple(sorted(mms))) for cr, mms in mcrs.items() ) )
        if key not in cmcc:
            cmcc[ key ] = []
        cmcc[ key ].append( kk )
    
    return cmcc

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

def parse_into_operation_tree( rx, ignoreCase ):
    
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
        
        if stack[-1][0] == 'character-class':
            if cc == ']' and len( stack[-1][1] ) != 0:
                mr = MatchCharacterClass( stack[-1][1], ignoreCase = ignoreCase )
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
                stack.append( ('character-class',[]) )
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
                matchType = MatchChar( cc, ignoreCase = ignoreCase )
            
            # special characters
            # 
            elif cc == 't':
                matchType = MatchChar( '\t', ignoreCase = ignoreCase )
            elif cc == 'n':
                matchType = MatchChar( '\n', ignoreCase = ignoreCase )
            elif cc == 'r':
                matchType = MatchChar( '\r', ignoreCase = ignoreCase )
            elif cc == 'f':
                matchType = MatchChar( '\f', ignoreCase = ignoreCase )
            elif cc == 'v':
                matchType = MatchChar( '\v', ignoreCase = ignoreCase )
            
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
                matchType = MatchChar( cc, ignoreCase = ignoreCase )
        
        stack[-1][1].append( matchType )
    
    # TODO, check stack is only top and last run to shove into it
    if len( stack ) == 2:
        ss = stack.pop()
        stack[0][1].append( MatchRun( ss[1] ) )
        return MatchAlternation( stack[0][1] )
    else:
        raise Exception( 'didnt close something' )

##

class MatchCharacterClass():
    def __init__( self, spec, ignoreCase ):
        self._spec       = spec
        self._ignoreCase = ignoreCase
        
        spec = spec[:]
        
        self._inverted = False
        self._cc       = set()
        
        if not spec:
            return
        
        if spec[0] == '^':
            self._inverted = True
            spec.pop(0)
        
        dangler     = None
        pendingDash = False
        while spec:
            cc = spec.pop(0)
            if cc == '-':
                if dangler == None:
                    self._cc.add( '-' )
                else:
                    pendingDash = True
            elif pendingDash:
                for ii in range( min( ord(dangler), ord(cc) ), max( ord(dangler), ord(cc) ) + 1 ):
                    self._cc.add( chr(ii) )
                dangler = None
                pendingDash = None
            else:
                dangler = cc
        
        if dangler != None:
            self._cc.add( dangler )
        
        if pendingDash:
            self._cc.add( pendingDash )
        
        return
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )
    
    def matches( self ):
        if self._inverted:
            for ii in range( 256 ):
                cc = chr( ii )
                if self._ignoreCase:
                    if (cc.upper() not in self._cc) and (cc.lower() not in self._cc):
                        yield cc.upper()
                        yield cc.lower()
                else:
                    if cc not in self._cc:
                        yield cc
        else:
            for cc in self._cc:
                if self._ignoreCase:
                    yield cc.upper()
                    yield cc.lower()
                else:
                    yield cc

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
    def __init__( self, cc, ignoreCase ):
        self._cc         = cc
        self._ignoreCase = ignoreCase
        return
    
    def __repr__( self ):
        return '<MatchChar %s>' % repr( self._cc )
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )
    
    def matches( self ):
        if self._ignoreCase:
            yield self._cc.upper()
            yield self._cc.lower()
        else:
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
    
    parser.add_option(
        '-i', '--ignore-case',
        dest    = 'ignoreCase',
        action  = 'store_true' ,
        default = False ,
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
