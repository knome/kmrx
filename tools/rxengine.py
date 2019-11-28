#!/usr/bin/env python3

# 
# compiles non-capturing regular expression into an efficient form
# 

C_HEADER_TEMPLATE = """\

#ifndef H_%(templateName)s
#define H_%(templateName)s

#include <stdio.h>
#include <stdint.h>

struct kmRx_%(rxName)s {
  uint%(chunkSize)s_t chunks [ %(numChunks)s ] ;
};

// reset the rx to its initial state, prior to input
// 
static
inline
void
kmRx_%(rxName)s__reset(
  struct kmRx_%(rxName)s * rx
){
  *rx = (struct kmRx_%(rxName)s) {{0}};
  rx->chunks[0] |= 1 ;
}

// whether the rx is currently in a matching state
// 
static
inline
int
kmRx_%(rxName)s__matches(
  struct kmRx_%(rxName)s * rx
){
  uint%(chunkSize)s_t * chunks = &rx->chunks[0] ;
  uint64_t oredEndBits = %(endChecks)s ;
  int matched = !! oredEndBits ;
  return matched ;
}

// process the given character, updating internal state
// 
static
inline
void
kmRx_%(rxName)s__step(
  struct kmRx_%(rxName)s * rx ,
  char cc
){
  uint%(chunkSize)s_t * chunks = &rx->chunks[0] ;
  %(thatIsASweetSwitchStatementYouMightSay)s
}

// print out the internal states
// 

static
inline
void
kmRx_%(rxName)s__debug(
  struct kmRx_%(rxName)s *
) __attribute__((unused)) ;

static
inline
void
kmRx_%(rxName)s__debug(
  struct kmRx_%(rxName)s * rx
){
  uint%(chunkSize)s_t chunkno = 0 ;
  while( chunkno < %(numChunks)s ){
    unsigned char bit = 0 ;
    fprintf( stderr, "chunks:\\n" );
    while( bit < %(chunkSize)su ){
      fprintf( stderr, " %%d", !! ( rx->chunks[ chunkno ] & ( 1ull << bit ) ) );
      bit ++ ;
    }
    fprintf( stderr, "\\n" );
    chunkno ++ ;
  }
}

#endif
// H_%(templateName)s

"""

C_GREP_TEMPLATE = """

// this is not grep

// invoke with a file name to try to mmap the file and print lines with compiled in regex
// invoke without a file to print lines on stdin with compiled in regex

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <inttypes.h>

%(cHeaderTemplate)s

#define USE(x) do{(void)(x);}while(0)

#define KMRX_PAGE_SIZE (4096)
#define KMRX_BUFFER_SIZE (KMRX_PAGE_SIZE * 64)

// strings longer that buffer size in stdio/read mode will be dropped

int scan_files_mmap( char ** );
int scan_files_read( char ** );
int scan_stdio();
int scan_files_common( int ) ; // fd

int
main(
  int     argc ,
  char ** argv
){
  if( argc - 1 == 1 ){
    return scan_files_mmap( argv + 1 );
  } else if( argc - 1 == 2 ){
    return scan_files_read( argv + 1 );
  } else if ( argc - 1 == 0 ){
    return scan_stdio();
  } else {
    fprintf( stderr, "bad args\\n" );
    return 1 ;
  }
  
  return 0 ;
}

int scan_files_mmap( char ** argv ){
  char * filename = argv[0] ;
  
  int fd = open( filename, O_RDONLY );
  if( fd < 0 ){
    fprintf( stderr, "could not open %%s : %%d : %%s\\n", filename, errno, strerror( errno ) );
    exit( 1 );
  }
  
  struct stat out ;
  if( fstat( fd, &out ) < 0 ){
    fprintf( stderr, "could not stat %%s : %%d : %%s\\n", filename, errno, strerror( errno ) );
    exit( 1 );
  }
  
  char * start = mmap( NULL, out.st_size, PROT_READ, MAP_PRIVATE, fd, 0 );
  if( start == MAP_FAILED ){
    fprintf( stderr, "could not mmap %%s : %%d : %%s\\n", filename, errno, strerror( errno ) );
    exit( 1 ) ;
  }
  
  if( madvise( start, out.st_size, MADV_SEQUENTIAL ) < 0 ){
    fprintf( stderr, "could not madvise mmaped memory for %%s : %%d : %%s\\n", filename, errno, strerror( errno ) );
  }
  
  ////
  
  struct kmRx_%(rxName)s state ;
  kmRx_%(rxName)s__reset( & state );
  
  if( out.st_size < 0 ){
    fprintf( stderr, "that is not possible\\n" );
    exit( 1 );
  }
  
  uint64_t filesize = (uint64_t) out.st_size ;
  char *   head     = start                  ;
  size_t   offset   = 0u                     ;
  
  int success = 0 ;
  
  while( offset < filesize ){
    if( start[ offset ] == '\\n' || start[ offset ] == '\\n' ){
      if( kmRx_%(rxName)s__matches( & state ) ){
        success = 1 ;
        size_t span = &start[offset] - head ;
        printf( "%%.*s\\n", (int) span, head );
      }
      offset ++ ;
      head = &start[offset];
      kmRx_%(rxName)s__reset( & state );
    } else {
      kmRx_%(rxName)s__step( & state, start[ offset ] );
      offset ++ ;
    }
  }
  
  fflush( stdout );
  
  // int 1+=success/0=failure --> exitcode 0=success/1=failure
  return ! success ;
}

int scan_stdio(){
  return scan_files_common( 1 );
}

int scan_files_read( char ** argv ){
  char * filename = argv[0] ;
  
  int fd = open( filename, O_RDONLY );
  if( fd < 0 ){
    fprintf( stderr, "could not open %%s : %%d : %%s\\n", filename, errno, strerror( errno ) );
    exit( 1 ) ;
  }
  
  return scan_files_common( fd );
}

int scan_files_common( int fd ){
  
  struct kmRx_%(rxName)s state ;
  kmRx_%(rxName)s__reset( & state );
  
  char buffer [ KMRX_BUFFER_SIZE ];
  ssize_t start    = 0 ; // start of current buffer data
  ssize_t end      = 0 ; // end of current buffer data
  int dropping     = 0 ; // whether we have started dropping data because a string was too large
  int success      = 0 ; // whether we have found any matching strings
  
  while(1){
    
    if( start == end ){
      start = end = 0 ;
    }
    
    ssize_t amountToRead =
       ( start == end
         ? KMRX_BUFFER_SIZE
         : ( end > start
             ? KMRX_BUFFER_SIZE - end
             : start - end
           )
       );
    
    ssize_t amount = read( fd, &buffer[end], amountToRead );
    if( amount < 0 ){
      fprintf( stderr, "error reading file : %%d : %%s\\n", errno, strerror( errno ) );
      exit( 1 );
    }
    
    if( amount == 0 ){
      // eof
      fflush( stdout );
      return ! success ;
    }
    
    ssize_t offset = start ;
    
    end += amount ;
    if( end == KMRX_BUFFER_SIZE ){
      end = 0 ;
    }
    
    int startDropping = start == end ;
    
    do {
      
      if( buffer[ offset ] == '\\n' || buffer[ offset ] == '\\r' ){
        
        if( (! dropping) && kmRx_%(rxName)s__matches( & state ) ){
          success = 1 ;
          if( offset < start ){
            ssize_t toend = &buffer[ KMRX_BUFFER_SIZE ] - &buffer[start] ;
            ssize_t tooff = &buffer[ offset ] - &buffer[0] ;
            
            printf( "%%.*s"   , (int) toend, &buffer[start] );
            printf( "%%.*s\\n", (int) tooff, buffer );
          } else {
            ssize_t span = offset - start ;
            
            printf( "%%.*s\\n", (int) span, &buffer[start] );
          }
        }
        
        offset ++ ;
        startDropping = 0 ;
        dropping = 0 ;
        if( offset == KMRX_BUFFER_SIZE ){ 
          offset = 0 ;
          start  = 0 ;
        } else {
          start  = offset ;
        }
        
        kmRx_%(rxName)s__reset( & state );
        
      } else {
        kmRx_%(rxName)s__step( & state, buffer[ offset ] );
        
        offset ++ ;
        if( offset == KMRX_BUFFER_SIZE ){
          offset = 0 ;
        }
      }
    } while( offset != end );
    
    if( (! dropping) && startDropping ){
      fprintf( stderr, "line too long\\n" );
      dropping = 1 ;
    }
    
  }
}

"""

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
    
    # if we have [0] -*-> [1] -a-> [2]
    #    we make [0] -a-> [2]
    # 
    backpropagate_free_transitions( transitions )
    
    # we used a special connection to mark transitions to success
    # they are backpropagated along with the other connections above
    # we now instruct the nodes to consume them and mark themselves as end states
    # 
    mark_ends( transitions )
    
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
    
    ends = extract_ends( transitions )
    
    if options.debug:
        debug( 'ends', ends )
    
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
    
    # unified mask-combinations
    # 
    umc = coalesce_grouped_characters_with_equal_transition_sets( mcc, mcu )
    
    if options.debug:
        debug()
        debug( 'umc', umc )
    
    switchStatement = that_is_a_sweet_switch_statement_you_might_say(
        maxChunk  = maxChunk          ,
        chunkSize = options.chunkSize ,
        umc       = umc               ,
    )
    
    if options.grepish:
        endChecks = generate_end_checks( options.chunkSize, ends )
        
        header = C_HEADER_TEMPLATE % {
            'chunkSize'                              : str( options.chunkSize) ,
            'rxName'                                 : options.name            ,
            'templateName'                           : options.template        ,
            'endChecks'                              : endChecks               ,
            'thatIsASweetSwitchStatementYouMightSay' : switchStatement         ,
            'numChunks'                              : maxChunk + 1            ,
        }
        
        template = C_GREP_TEMPLATE % {
            'chunkSize'       : str( options.chunkSize ) ,
            'rxName'          : options.name ,
            'cHeaderTemplate' : header ,
        }
        
        print( template )
        
    elif options.libraryHeader:
        endChecks = generate_end_checks( options.chunkSize, ends )
        
        header = C_HEADER_TEMPLATE % {
            'chunkSize'                              : str( options.chunkSize) ,
            'rxName'                                 : options.name            ,
            'templateName'                           : options.template        ,
            'endChecks'                              : endChecks               ,
            'thatIsASweetSwitchStatementYouMightSay' : switchStatement         ,
            'numChunks'                              : maxChunk + 1            ,
        }
        
        print( header )
        
    else:
        print( switchStatement )
    
    return

##

def that_is_a_sweet_switch_statement_you_might_say( maxChunk, chunkSize, umc ):
    # if it was still a switch statement
    
    bits = []
    
    # 
    # build a jump table
    # 
    
    jumps = dict(
        ( ii, None )
        for ii in range( 256 )
    )
    
    enumeratedTransitions = {}
    defaultJump = 'clear'
    for eno, (transitions, kks) in enumerate( umc.items() ):
        enumeratedTransitions[ eno ] = transitions
        for kk in kks:
            if kk != None:
                jumps[ ord(kk) ] = 'jump_%s' % str( eno )
            else:
                defaultJump = 'jump_%s' % str( eno )
    
    for ii in jumps:
        if jumps[ ii ] == None:
            jumps[ ii ] = defaultJump
    
    bits.append( ' static void * const jumpTable [ 256 ] = { ' )
    bits.append( ','.join(
        ' && %s' % vv
        for ii, vv in sorted( jumps.items() )
        )
    )
    bits.append( ' };\n' )
    
    # bits.append( ' fprintf( stderr, "going to %u\\n", (unsigned char) cc );\n' )
    bits.append( ' goto *jumpTable[ (unsigned char) cc ];\n' )
    
    # 
    # build the targets
    # 
    
    if 'clear' in jumps.values():
        bits.append( 'clear: {\n' )
        for ii in range( maxChunk + 1 ):
            bits.append( '    chunks[ %s ] = 0 ;\n' % ii )
        bits.append( '  return;\n' )
        bits.append( '}\n')
    
    for eno, mcrs in enumeratedTransitions.items():
        kks = umc[ mcrs ]
        
        # generate jump target
        # 
        bits.append( 'jump_%s:{\n' % eno )
        
        # 
        # generate shifty bits
        # 
        
        # read bits before we overwrite them
        # 
        for chunk in set( cr[0] for (cr,_) in mcrs ):
            bits.append(
                '    uint%(chunkSize)s_t prev_%(chunk)s = chunks[ %(chunk)s ];\n' % {
                    'chunkSize' : chunkSize ,
                    'chunk'     : chunk     ,
                }
            )
        
        # overwrite them
        # 
        for ii in range( maxChunk + 1 ):
            bits.append( '    chunks[ %s ] = 0 ;\n' % ii )
        
        # write updated bits
        # 
        for cr, mms in mcrs:
            bits.append( '    chunks[ %s ] |= ( (prev_%s & %s) %s %s ); // %s\n' % (
                cr[2] ,
                cr[0] ,
                hex( sum( (1 << mm) for mm in mms ) ) + 'ull' ,
                '>>' if cr[1] < 0  else '<<' ,
                -cr[1] if cr[1] < 0 else cr[1] ,
                ', '.join( '(%s,%s)' % (mm,mm+cr[1]) for mm in mms) ,
            ))
        
        # jump done
        # 
        bits.append( '  return ;\n' )
        bits.append( '  }\n' )
    
    return ''.join( bits )

##

def coalesce_grouped_characters_with_equal_transition_sets(
    mcc ,
    mcu ,
):
    umc = {}
    for kk, mcrs in list( mcc.items() ) + [ (None, mcu), ]:
        key = tuple( sorted( (cr, tuple(sorted(mms))) for cr, mms in mcrs.items() ) )
        if key not in umc:
            umc[ key ] = []
        umc[ key ].append( kk )
    
    return umc

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
                combos[ key ] = set()
            combos[ key ].add( im )
        outc[ kk ] = combos
    
    combos = {}
    for transition in cmrUTriggers:
        ic, im, ir, cc = transition
        key = (ic, ir, cc)
        if key not in combos:
            combos[ key ] = set()
        combos[ key ].add( im )
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

def mark_ends( nodes ):
    for node in nodes:
        node.mark_if_end()
    return

##

def extract_ends( nodes ):
    out = []
    for node in nodes:
        if node.is_end():
            out.append( node.index() )
    return out

##

def generate_end_checks( chunkSize, ends ):
    bits = []
    for end in ends:
        bits.append(
            ' (chunks[ %s ] & %sull) ' % (
                end // chunkSize ,
                1 << (end % chunkSize) ,
            )
        )
    return ' ( ' + ' | '.join(bits) + ' ) '

## 



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
    
    operationTree.create_and_thread_nodes( start, stop )
    stop.connect( True, end )
    
    return start, end

class Node():
    def __init__( self ):
        self._outs  = []
        self._index = None
        self._end   = False
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
    
    def is_end( self ):
        return self._end
    
    def mark_if_end( self ):
        # (True, Node) links will have been dragged back from the terminal node
        # use them to determine if advancing to this node indicates an end scenario
        # we can then have anything that performs this transition
        
        fixed = []
        for out in self._outs:
            if out[0] == True:
                self._end = True
            else:
                fixed.append( out )
        
        self._outs = fixed
    
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
            if cc == '}':
                # get element we were building
                rep = stack.pop()
                # get previous matcher in current run
                if not stack[-1][1]: raise Exception( 'dangling {}' )
                prev = stack[-1][1].pop()
                # create new repetition matcher
                mr = MatchRepetition( rep[1], prev )
                # slide it back into the run
                stack[-1][1].append( mr )
                continue
            else:
                stack[-1][1].append( cc )
                continue
        
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
                stack.append( ('repetition', []) )
                continue
            
            if cc == '}':
                raise Exception( 'bare } outside repetition {...}' )
        
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
            elif cc == '\\':
                matchType = MatchChar( '\\', ignoreCase = ignoreCase )
            
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
                raise Exception( 'unknown escape match : %s' % repr( cc ) )
            
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
                if dangler != None:
                    self._cc.add( dangler )
                dangler = cc
        
        if dangler != None:
            self._cc.add( dangler )
        
        if pendingDash:
            self._cc.add( '-' )
        
        return
    
    def __repr__( self ):
        return '<MatchCharacterClass %s>' % repr( self._spec )
    
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


class MatchRepetition():
    def __init__( self, spec, mm ):
        self._spec = spec
        self._mm = mm
        
        chunks = ''.join( spec ).split(',')
        
        if len( chunks ) == 1:
            chunk = chunks[0].strip()
            if chunk == '':
                self._min = 0
                self._max = None
            elif chunk.isdigit():
                self._min = int(chunk, 10)
                self._max = int(chunk, 10)
            else:
                raise Exception( 'garbage in {} : %s' % repr( spec ) )
        
        elif len( chunks ) == 2:
            first = chunks[0].strip()
            if first == '':
                self._min = 0
            elif first.isdigit():
                self._min = int(first,10)
            else:
                raise Exception( 'garbage in first slot of {} : %s' % repr( spec ) )
            
            second = chunks[1].strip()
            if second == '':
                self._max = None
            elif second.isdigit():
                self._max = int(second,10)
            else:
                raise Exception( 'garbage in second slot of {} : %s' % repr( spec ) )
            
            if self._max != None:
                if self._min > self._max:
                    raise Exception( 'cannot have bigger min than max in {} : %s' % repr( spec ) )
            
        else:
            raise Exception( 'too many slots in {} : %s' % repr( spec ) )
    
    def create_and_thread_nodes( self, start, stop ):
        # if current (1-index) > min, we have to let a bypass
        # if max we have to thread through max nodes
        # if max == None we have to thread through min nodes and then * out
        # if min == 0 and max == None, we're literally a *
        # if min == 0 and max == 0, we're a comment
        
        previous = start
        for _ in range( self._min ):
            node = Node()
            self._mm.create_and_thread_nodes( previous, node )
            previous = node
        
        if self._max != None:
            for _ in range( self._min, self._max ):
                node = Node()
                self._mm.create_and_thread_nodes( previous, node )
                previous.connect( None, stop )
                previous = node
            previous.connect( None, stop )
        else:
            self._mm.create_and_thread_nodes( previous, previous )
            previous.connect( None, stop )
        
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
            enterNode = Node()
            start.connect( None, enterNode )
            run.create_and_thread_nodes( enterNode, stop )
        return

class MatchDigit():
    def __repr__( self ):
        return '<MatchDigit>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )
    
    def matches( self ):
        for cc in '0123456789':
            yield cc
    
class MatchNotDigit():
    def __repr__( self ):
        return '<MatchNotDigit>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )
    
    def matches( self ):
        for ii in range( 256 ):
            cc = chr( ii )
            if cc not in '0123456789':
                yield cc
    
class MatchWhitespace():
    def __repr__( self ):
        return '<MatchWhitespace>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )
    
    def matches( self ):
        for cc in ' \t\n\r\f\v':
            yield cc
    
class MatchNotWhitespace():
    def __repr__( self ):
        return '<MatchNotWhitespace>'
    
    def create_and_thread_nodes( self, start, stop ):
        start.connect( self, stop )
    
    def matches( self ):
        for ii in range( 256 ):
            cc = chr( ii )
            if cc not in ' \t\n\r\f\v':
                yield cc

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
    
    # regex options
    
    parser.add_option(
        '-c', '--chunkSize',
        dest    = 'chunkSize' ,
        type    = 'int' ,
        default = 64 ,
        help    = 'size to use for chunking out state data (8,16,32,64,128)'
    )
    
    parser.add_option(
        '-i', '--ignore-case',
        dest    = 'ignoreCase',
        action  = 'store_true' ,
        default = False ,
    )
    
    # generator options
    
    parser.add_option(
        '-n', '--name',
        dest    = 'name',
        default = 'KmrxExample',
        help    = 'a prefix for the regex bits',
    )
    
    parser.add_option(
        '-t', '--template',
        dest    = 'template',
        default = 'KMRX_EXAMPLE',
        help    = 'a term to use in macros',
    )
    
    parser.add_option(
        '-l', '--library-header',
        action = 'store_true',
        dest = 'libraryHeader',
        help = '-h wouldnt exactly work, would it? make the header',
    )
    
    parser.add_option(
        '-g', '--grepish',
        dest   = 'grepish',
        action = 'store_true',
        help   = 'output a grep-like program using the regex',
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
