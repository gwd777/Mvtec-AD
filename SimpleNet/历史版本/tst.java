package hump_yard_4;

import java.io.Serializable;
import java.sql.Connection;
import java.sql.SQLException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Currency;
import java.util.Date;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.Stack;
import java.util.Timer;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Vector;
import java.awt.Color;
import java.awt.Font;
import com.anylogic.engine.connectivity.ResultSet;
import com.anylogic.engine.connectivity.Statement;
import com.anylogic.engine.elements.*;
import com.anylogic.engine.markup.Network;
import com.anylogic.engine.Position;
import com.anylogic.engine.markup.PedFlowStatistics;
import com.anylogic.engine.markup.DensityMap;


import static java.lang.Math.*;
import static com.anylogic.engine.UtilitiesArray.*;
import static com.anylogic.engine.UtilitiesCollection.*;
import static com.anylogic.engine.presentation.UtilitiesColor.*;
import static com.anylogic.engine.HyperArray.*;

import com.anylogic.engine.*;
import com.anylogic.engine.analysis.*;
import com.anylogic.engine.connectivity.*;
import com.anylogic.engine.database.*;
import com.anylogic.engine.gis.*;
import com.anylogic.engine.markup.*;
import com.anylogic.engine.routing.*;
import com.anylogic.engine.presentation.*;
import com.anylogic.engine.gui.*;

import com.anylogic.libraries.rail.*;
import com.anylogic.libraries.processmodeling.*;
import com.anylogic.libraries.modules.markup_descriptors.*;

import java.awt.geom.Arc2D;

public class Main extends Agent
{
  // 参数
  // 简单变量

 public RailwayTrack trackReadyToDepart;

  @AnyLogicInternalCodegenAPI
  private static Map<String, IElementDescriptor> elementDesciptors_xjal = createElementDescriptors( Main.class );

  @AnyLogicInternalCodegenAPI
  @Override
  public Map<String, IElementDescriptor> getElementDesciptors() {
    return elementDesciptors_xjal;
  }

  @AnyLogicCustomProposalPriority(type = AnyLogicCustomProposalPriority.Type.STATIC_ELEMENT)
  public static final Scale scale = new Scale( 1.0 );

  @Override
  public Scale getScale() {
    return scale;
  }





  /** Internal constant, shouldn't be accessed by user */
  @AnyLogicInternalCodegenAPI
  protected static final int _STATECHART_COUNT_xjal = 0;


  // 嵌入对象

  public com.anylogic.libraries.rail.TrainSource<Train,Agent> trainSource;
  public com.anylogic.libraries.rail.TrainMoveTo<Train> trainMoveTo;
  public com.anylogic.libraries.rail.TrainDecouple<Train,Agent> trainDecouple;
  public com.anylogic.libraries.rail.TrainCouple<Train,Agent> trainCouple;
  public com.anylogic.libraries.rail.TrainDispose<Train> trainDispose;
  public com.anylogic.libraries.rail.TrainMoveTo<Agent> trainMoveTo1;
  public com.anylogic.libraries.rail.TrainMoveTo<Agent> trainMoveTo2;
  public com.anylogic.libraries.rail.TrainMoveTo<Agent> trainMoveTo3;
  public com.anylogic.libraries.rail.TrainMoveTo<Train> trainMoveTo4;
  public com.anylogic.libraries.processmodeling.Delay<Train> delay;
  public com.anylogic.libraries.rail.TrainDecouple<Train,Train> trainDecouple1;
  public com.anylogic.libraries.processmodeling.Delay<Train> delay1;
  public com.anylogic.libraries.processmodeling.SelectOutput<Train> selectOutput;
  public com.anylogic.libraries.rail.TrainMoveTo<Train> trainMoveTo5;
  public com.anylogic.libraries.rail.TrainMoveTo<Train> trainMoveTo6;
  public com.anylogic.libraries.rail.TrainCouple<Train,Train> trainCouple1;
  public com.anylogic.libraries.processmodeling.SelectOutput<Train> selectOutput1;
  public com.anylogic.libraries.rail.TrainCouple<Train,Train> trainCouple2;
  public com.anylogic.libraries.rail.TrainMoveTo<Train> trainMoveTo7;
  public com.anylogic.libraries.rail.TrainDispose<Train> trainDispose1;
  public com.anylogic.libraries.rail.TrainMoveTo<Train> trainMoveTo8;
  public com.anylogic.libraries.rail.TrainSource<Train,Agent> newLoco;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineN8_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineN7_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineN5_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineN4_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineN3_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineN6_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineN1_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineN2_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineN11_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineN12_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineN9_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineN10_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineHump_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineArrival_controller_xjal;
  public com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _stopLineEntry_controller_xjal;

  public String getNameOf( Agent ao ) {
    if ( ao == trainSource ) return "trainSource";
    if ( ao == trainMoveTo ) return "trainMoveTo";
    if ( ao == trainDecouple ) return "trainDecouple";
    if ( ao == trainCouple ) return "trainCouple";
    if ( ao == trainDispose ) return "trainDispose";
    if ( ao == trainMoveTo1 ) return "trainMoveTo1";
    if ( ao == trainMoveTo2 ) return "trainMoveTo2";
    if ( ao == trainMoveTo3 ) return "trainMoveTo3";
    if ( ao == trainMoveTo4 ) return "trainMoveTo4";
    if ( ao == delay ) return "delay";
    if ( ao == trainDecouple1 ) return "trainDecouple1";
    if ( ao == delay1 ) return "delay1";
    if ( ao == selectOutput ) return "selectOutput";
    if ( ao == trainMoveTo5 ) return "trainMoveTo5";
    if ( ao == trainMoveTo6 ) return "trainMoveTo6";
    if ( ao == trainCouple1 ) return "trainCouple1";
    if ( ao == selectOutput1 ) return "selectOutput1";
    if ( ao == trainCouple2 ) return "trainCouple2";
    if ( ao == trainMoveTo7 ) return "trainMoveTo7";
    if ( ao == trainDispose1 ) return "trainDispose1";
    if ( ao == trainMoveTo8 ) return "trainMoveTo8";
    if ( ao == newLoco ) return "newLoco";
    if ( ao == _stopLineN8_controller_xjal ) return "_stopLineN8_controller_xjal";
    if ( ao == _stopLineN7_controller_xjal ) return "_stopLineN7_controller_xjal";
    if ( ao == _stopLineN5_controller_xjal ) return "_stopLineN5_controller_xjal";
    if ( ao == _stopLineN4_controller_xjal ) return "_stopLineN4_controller_xjal";
    if ( ao == _stopLineN3_controller_xjal ) return "_stopLineN3_controller_xjal";
    if ( ao == _stopLineN6_controller_xjal ) return "_stopLineN6_controller_xjal";
    if ( ao == _stopLineN1_controller_xjal ) return "_stopLineN1_controller_xjal";
    if ( ao == _stopLineN2_controller_xjal ) return "_stopLineN2_controller_xjal";
    if ( ao == _stopLineN11_controller_xjal ) return "_stopLineN11_controller_xjal";
    if ( ao == _stopLineN12_controller_xjal ) return "_stopLineN12_controller_xjal";
    if ( ao == _stopLineN9_controller_xjal ) return "_stopLineN9_controller_xjal";
    if ( ao == _stopLineN10_controller_xjal ) return "_stopLineN10_controller_xjal";
    if ( ao == _stopLineHump_controller_xjal ) return "_stopLineHump_controller_xjal";
    if ( ao == _stopLineArrival_controller_xjal ) return "_stopLineArrival_controller_xjal";
    if ( ao == _stopLineEntry_controller_xjal ) return "_stopLineEntry_controller_xjal";
    return super.getNameOf( ao );
  }

  public AgentAnimationSettings getAnimationSettingsOf( Agent ao ) {
    return super.getAnimationSettingsOf( ao );
  }


  public String getNameOf( AgentList<?> aolist ) {
    return super.getNameOf( aolist );
  }

  public AgentAnimationSettings getAnimationSettingsOf( AgentList<?> aolist ) {
    return super.getAnimationSettingsOf( aolist );
  }


  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainSource<Train, Agent> instantiate_trainSource_xjal() {
    com.anylogic.libraries.rail.TrainSource<Train, Agent> _result_xjal = new com.anylogic.libraries.rail.TrainSource<Train, Agent>( getEngine(), this, null ) {
      @Override
      public double interarrivalTime(  ) {
        return _trainSource_interarrivalTime_xjal( this );
      }

      @AnyLogicInternalCodegenAPI
      public TimeUnits getUnitsForCodeOf_interarrivalTime() {
        return MINUTE;
      }
      @Override
      public RailwayTrack track( Train train ) {
        return _trainSource_track_xjal( this, train );
      }
      @Override
      public Agent newTrain(  ) {
        return _trainSource_newTrain_xjal( this );
      }
      @Override
      public Agent newRailCar( Train train, int carindex ) {
        return _trainSource_newRailCar_xjal( this, train, carindex );
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119373596715L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainSource_xjal( final com.anylogic.libraries.rail.TrainSource<Train, Agent> self, TableInput _t ) {
    self.arrivalType = self._arrivalType_DefaultValue_xjal();
    self.firstArrivalMode = self._firstArrivalMode_DefaultValue_xjal();
    self.firstArrivalTime = self._firstArrivalTime_DefaultValue_xjal();
    self.arrivalSchedule = self._arrivalSchedule_DefaultValue_xjal();
    self.setAgentParametersFromDB = self._setAgentParametersFromDB_DefaultValue_xjal();
    self.databaseTable = self._databaseTable_DefaultValue_xjal();
    self.limitArrivals = self._limitArrivals_DefaultValue_xjal();
    self.maxArrivals = self._maxArrivals_DefaultValue_xjal();
    self.putInRailYard = self._putInRailYard_DefaultValue_xjal();
    self.locationType =
self.LOCATION_TRACK_OFFSET
;
    self.addTrainToCustomPopulation = self._addTrainToCustomPopulation_DefaultValue_xjal();
    self.addCarToCustomPopulation = self._addCarToCustomPopulation_DefaultValue_xjal();
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainSource_xjal( com.anylogic.libraries.rail.TrainSource<Train, Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainMoveTo<Train> instantiate_trainMoveTo_xjal() {
    com.anylogic.libraries.rail.TrainMoveTo<Train> _result_xjal = new com.anylogic.libraries.rail.TrainMoveTo<Train>( getEngine(), this, null ) {
      @Override
      public PositionOnTrack<?> positionOnTrack( Train train ) {
        return _trainMoveTo_positionOnTrack_xjal( this, train );
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119373581434L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainMoveTo_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, TableInput _t ) {
    self.forward = self._forward_DefaultValue_xjal();
    self.routeType =
self.ROUTE_AUTO
;
    self.BlockedTrackHandling = self._BlockedTrackHandling_DefaultValue_xjal();
    self.targetType =
self.TARGET_AT_POINT_ON_TRACK
;
    self.checkFreeSpaceOnTargetTrack = self._checkFreeSpaceOnTargetTrack_DefaultValue_xjal();
    self.limitDistance = self._limitDistance_DefaultValue_xjal();
    self.distanceIs = self._distanceIs_DefaultValue_xjal();
    self.startOptions = self._startOptions_DefaultValue_xjal();
    self.finishOptions =
self.ACCELERATE_YES
;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainMoveTo_xjal( com.anylogic.libraries.rail.TrainMoveTo<Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainDecouple<Train, Agent> instantiate_trainDecouple_xjal() {
    com.anylogic.libraries.rail.TrainDecouple<Train, Agent> _result_xjal = new com.anylogic.libraries.rail.TrainDecouple<Train, Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainDecouple_xjal( final com.anylogic.libraries.rail.TrainDecouple<Train, Agent> self, TableInput _t ) {
    self.decoupleFirstCars = self._decoupleFirstCars_DefaultValue_xjal();
    self.addToCustomPopulation = self._addToCustomPopulation_DefaultValue_xjal();
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainDecouple_xjal( com.anylogic.libraries.rail.TrainDecouple<Train, Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainCouple<Train, Agent> instantiate_trainCouple_xjal() {
    com.anylogic.libraries.rail.TrainCouple<Train, Agent> _result_xjal = new com.anylogic.libraries.rail.TrainCouple<Train, Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainCouple_xjal( final com.anylogic.libraries.rail.TrainCouple<Train, Agent> self, TableInput _t ) {
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainCouple_xjal( com.anylogic.libraries.rail.TrainCouple<Train, Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainDispose<Train> instantiate_trainDispose_xjal() {
    com.anylogic.libraries.rail.TrainDispose<Train> _result_xjal = new com.anylogic.libraries.rail.TrainDispose<Train>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainDispose_xjal( final com.anylogic.libraries.rail.TrainDispose<Train> self, TableInput _t ) {
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainDispose_xjal( com.anylogic.libraries.rail.TrainDispose<Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainMoveTo<Agent> instantiate_trainMoveTo1_xjal() {
    com.anylogic.libraries.rail.TrainMoveTo<Agent> _result_xjal = new com.anylogic.libraries.rail.TrainMoveTo<Agent>( getEngine(), this, null ) {
      @Override
      public PositionOnTrack<?> positionOnTrack( Agent train ) {
        return _trainMoveTo1_positionOnTrack_xjal( this, train );
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119374891391L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainMoveTo1_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Agent> self, TableInput _t ) {
    self.forward = self._forward_DefaultValue_xjal();
    self.routeType =
self.ROUTE_AUTO
;
    self.BlockedTrackHandling = self._BlockedTrackHandling_DefaultValue_xjal();
    self.targetType =
self.TARGET_AT_POINT_ON_TRACK
;
    self.checkFreeSpaceOnTargetTrack = self._checkFreeSpaceOnTargetTrack_DefaultValue_xjal();
    self.limitDistance = self._limitDistance_DefaultValue_xjal();
    self.distanceIs = self._distanceIs_DefaultValue_xjal();
    self.startOptions = self._startOptions_DefaultValue_xjal();
    self.finishOptions =
self.ACCELERATE_YES
;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainMoveTo1_xjal( com.anylogic.libraries.rail.TrainMoveTo<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainMoveTo<Agent> instantiate_trainMoveTo2_xjal() {
    com.anylogic.libraries.rail.TrainMoveTo<Agent> _result_xjal = new com.anylogic.libraries.rail.TrainMoveTo<Agent>( getEngine(), this, null ) {
      @Override
      public RailwayTrack[] tracksToAvoid( Agent train ) {
        return _trainMoveTo2_tracksToAvoid_xjal( this, train );
      }
      @Override
      public PositionOnTrack<?> positionOnTrack( Agent train ) {
        return _trainMoveTo2_positionOnTrack_xjal( this, train );
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119374822458L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainMoveTo2_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Agent> self, TableInput _t ) {
    self.forward =
false
;
    self.routeType =
self.ROUTE_AUTO
;
    self.BlockedTrackHandling = self._BlockedTrackHandling_DefaultValue_xjal();
    self.targetType =
self.TARGET_AT_POINT_ON_TRACK
;
    self.checkFreeSpaceOnTargetTrack = self._checkFreeSpaceOnTargetTrack_DefaultValue_xjal();
    self.limitDistance = self._limitDistance_DefaultValue_xjal();
    self.distanceIs = self._distanceIs_DefaultValue_xjal();
    self.startOptions = self._startOptions_DefaultValue_xjal();
    self.finishOptions = self._finishOptions_DefaultValue_xjal();
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainMoveTo2_xjal( com.anylogic.libraries.rail.TrainMoveTo<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainMoveTo<Agent> instantiate_trainMoveTo3_xjal() {
    com.anylogic.libraries.rail.TrainMoveTo<Agent> _result_xjal = new com.anylogic.libraries.rail.TrainMoveTo<Agent>( getEngine(), this, null ) {
      @Override
      public PositionOnTrack<?> positionOnTrack( Agent train ) {
        return _trainMoveTo3_positionOnTrack_xjal( this, train );
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119553931322L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainMoveTo3_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Agent> self, TableInput _t ) {
    self.forward = self._forward_DefaultValue_xjal();
    self.routeType =
self.ROUTE_AUTO
;
    self.BlockedTrackHandling = self._BlockedTrackHandling_DefaultValue_xjal();
    self.targetType =
self.TARGET_AT_POINT_ON_TRACK
;
    self.checkFreeSpaceOnTargetTrack = self._checkFreeSpaceOnTargetTrack_DefaultValue_xjal();
    self.limitDistance = self._limitDistance_DefaultValue_xjal();
    self.distanceIs = self._distanceIs_DefaultValue_xjal();
    self.startOptions = self._startOptions_DefaultValue_xjal();
    self.finishOptions =
self.ACCELERATE_YES
;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainMoveTo3_xjal( com.anylogic.libraries.rail.TrainMoveTo<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainMoveTo<Train> instantiate_trainMoveTo4_xjal() {
    com.anylogic.libraries.rail.TrainMoveTo<Train> _result_xjal = new com.anylogic.libraries.rail.TrainMoveTo<Train>( getEngine(), this, null ) {
      @Override
      public PositionOnTrack<?> positionOnTrack( Train train ) {
        return _trainMoveTo4_positionOnTrack_xjal( this, train );
      }
      @Override
      public double cruiseSpeed( Train train ) {
        return _trainMoveTo4_cruiseSpeed_xjal( this, train );
      }

      @AnyLogicInternalCodegenAPI
      public SpeedUnits getUnitsForCodeOf_cruiseSpeed() {
        return MPS;
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119554213950L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainMoveTo4_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, TableInput _t ) {
    self.forward = self._forward_DefaultValue_xjal();
    self.routeType =
self.ROUTE_AUTO
;
    self.BlockedTrackHandling = self._BlockedTrackHandling_DefaultValue_xjal();
    self.targetType =
self.TARGET_AT_POINT_ON_TRACK
;
    self.checkFreeSpaceOnTargetTrack = self._checkFreeSpaceOnTargetTrack_DefaultValue_xjal();
    self.limitDistance = self._limitDistance_DefaultValue_xjal();
    self.distanceIs = self._distanceIs_DefaultValue_xjal();
    self.startOptions = self._startOptions_DefaultValue_xjal();
    self.finishOptions =
self.ACCELERATE_YES
;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainMoveTo4_xjal( com.anylogic.libraries.rail.TrainMoveTo<Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.processmodeling.Delay<Train> instantiate_delay_xjal() {
    com.anylogic.libraries.processmodeling.Delay<Train> _result_xjal = new com.anylogic.libraries.processmodeling.Delay<Train>( getEngine(), this, null ) {
      @Override
      public double delayTime( Train agent ) {
        return _delay_delayTime_xjal( this, agent );
      }

      @AnyLogicInternalCodegenAPI
      public TimeUnits getUnitsForCodeOf_delayTime() {
        return SECOND;
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119554194475L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_delay_xjal( final com.anylogic.libraries.processmodeling.Delay<Train> self, TableInput _t ) {
    self.type = self._type_DefaultValue_xjal();
    self.capacity = self._capacity_DefaultValue_xjal();
    self.maximumCapacity = self._maximumCapacity_DefaultValue_xjal();
    self.entityLocation = self._entityLocation_DefaultValue_xjal();
    self.pushProtocol = self._pushProtocol_DefaultValue_xjal();
    self.restoreEntityLocationOnExit = self._restoreEntityLocationOnExit_DefaultValue_xjal();
    self.forceStatisticsCollection = self._forceStatisticsCollection_DefaultValue_xjal();
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_delay_xjal( com.anylogic.libraries.processmodeling.Delay<Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainDecouple<Train, Train> instantiate_trainDecouple1_xjal() {
    com.anylogic.libraries.rail.TrainDecouple<Train, Train> _result_xjal = new com.anylogic.libraries.rail.TrainDecouple<Train, Train>( getEngine(), this, null ) {
      @Override
      public int nCars( Train train ) {
        return _trainDecouple1_nCars_xjal( this, train );
      }
      @Override
      public Agent newTrain(  ) {
        return _trainDecouple1_newTrain_xjal( this );
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119625251131L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainDecouple1_xjal( final com.anylogic.libraries.rail.TrainDecouple<Train, Train> self, TableInput _t ) {
    self.decoupleFirstCars = self._decoupleFirstCars_DefaultValue_xjal();
    self.addToCustomPopulation = self._addToCustomPopulation_DefaultValue_xjal();
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainDecouple1_xjal( com.anylogic.libraries.rail.TrainDecouple<Train, Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.processmodeling.Delay<Train> instantiate_delay1_xjal() {
    com.anylogic.libraries.processmodeling.Delay<Train> _result_xjal = new com.anylogic.libraries.processmodeling.Delay<Train>( getEngine(), this, null ) {
      @Override
      public double delayTime( Train agent ) {
        return _delay1_delayTime_xjal( this, agent );
      }

      @AnyLogicInternalCodegenAPI
      public TimeUnits getUnitsForCodeOf_delayTime() {
        return SECOND;
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119625251962L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_delay1_xjal( final com.anylogic.libraries.processmodeling.Delay<Train> self, TableInput _t ) {
    self.type = self._type_DefaultValue_xjal();
    self.capacity = self._capacity_DefaultValue_xjal();
    self.maximumCapacity = self._maximumCapacity_DefaultValue_xjal();
    self.entityLocation = self._entityLocation_DefaultValue_xjal();
    self.pushProtocol = self._pushProtocol_DefaultValue_xjal();
    self.restoreEntityLocationOnExit = self._restoreEntityLocationOnExit_DefaultValue_xjal();
    self.forceStatisticsCollection = self._forceStatisticsCollection_DefaultValue_xjal();
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_delay1_xjal( com.anylogic.libraries.processmodeling.Delay<Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.processmodeling.SelectOutput<Train> instantiate_selectOutput_xjal() {
    com.anylogic.libraries.processmodeling.SelectOutput<Train> _result_xjal = new com.anylogic.libraries.processmodeling.SelectOutput<Train>( getEngine(), this, null ) {
      @Override
      public boolean condition( Train agent ) {
        return _selectOutput_condition_xjal( this, agent );
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119625435439L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_selectOutput_xjal( final com.anylogic.libraries.processmodeling.SelectOutput<Train> self, TableInput _t ) {
    self.conditionIsProbabilistic =
false
;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_selectOutput_xjal( com.anylogic.libraries.processmodeling.SelectOutput<Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainMoveTo<Train> instantiate_trainMoveTo5_xjal() {
    com.anylogic.libraries.rail.TrainMoveTo<Train> _result_xjal = new com.anylogic.libraries.rail.TrainMoveTo<Train>( getEngine(), this, null ) {
      @Override
      public RailwayTrack targetTrack( Train train ) {
        return _trainMoveTo5_targetTrack_xjal( this, train );
      }
      @Override
      public double cruiseSpeed( Train train ) {
        return _trainMoveTo5_cruiseSpeed_xjal( this, train );
      }

      @AnyLogicInternalCodegenAPI
      public SpeedUnits getUnitsForCodeOf_cruiseSpeed() {
        return MPS;
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119625497706L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainMoveTo5_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, TableInput _t ) {
    self.forward = self._forward_DefaultValue_xjal();
    self.routeType = self.ROUTE_AUTO;
    self.BlockedTrackHandling = self._BlockedTrackHandling_DefaultValue_xjal();
    self.targetType = self.TARGET_AT_TRACK_OFFSET;
    self.checkFreeSpaceOnTargetTrack = self._checkFreeSpaceOnTargetTrack_DefaultValue_xjal();
    self.limitDistance = self._limitDistance_DefaultValue_xjal();
    self.distanceIs = self._distanceIs_DefaultValue_xjal();
    self.startOptions = self._startOptions_DefaultValue_xjal();
    self.finishOptions = self._finishOptions_DefaultValue_xjal();
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainMoveTo5_xjal( com.anylogic.libraries.rail.TrainMoveTo<Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainMoveTo<Train> instantiate_trainMoveTo6_xjal() {
    com.anylogic.libraries.rail.TrainMoveTo<Train> _result_xjal = new com.anylogic.libraries.rail.TrainMoveTo<Train>( getEngine(), this, null ) {
      @Override
      public PositionOnTrack<?> positionOnTrack( Train train ) {
        return _trainMoveTo6_positionOnTrack_xjal( this, train );
      }
      @Override
      public double cruiseSpeed( Train train ) {
        return _trainMoveTo6_cruiseSpeed_xjal( this, train );
      }

      @AnyLogicInternalCodegenAPI
      public SpeedUnits getUnitsForCodeOf_cruiseSpeed() {
        return MPS;
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119625234751L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainMoveTo6_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, TableInput _t ) {
    self.forward = self._forward_DefaultValue_xjal();
    self.routeType =
self.ROUTE_AUTO
;
    self.BlockedTrackHandling = self._BlockedTrackHandling_DefaultValue_xjal();
    self.targetType =
self.TARGET_AT_POINT_ON_TRACK
;
    self.checkFreeSpaceOnTargetTrack = self._checkFreeSpaceOnTargetTrack_DefaultValue_xjal();
    self.limitDistance = self._limitDistance_DefaultValue_xjal();
    self.distanceIs = self._distanceIs_DefaultValue_xjal();
    self.startOptions = self._startOptions_DefaultValue_xjal();
    self.finishOptions =
self.ACCELERATE_YES
;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainMoveTo6_xjal( com.anylogic.libraries.rail.TrainMoveTo<Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainCouple<Train, Train> instantiate_trainCouple1_xjal() {
    com.anylogic.libraries.rail.TrainCouple<Train, Train> _result_xjal = new com.anylogic.libraries.rail.TrainCouple<Train, Train>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainCouple1_xjal( final com.anylogic.libraries.rail.TrainCouple<Train, Train> self, TableInput _t ) {
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainCouple1_xjal( com.anylogic.libraries.rail.TrainCouple<Train, Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.processmodeling.SelectOutput<Train> instantiate_selectOutput1_xjal() {
    com.anylogic.libraries.processmodeling.SelectOutput<Train> _result_xjal = new com.anylogic.libraries.processmodeling.SelectOutput<Train>( getEngine(), this, null ) {
      @Override
      public boolean condition( Train agent ) {
        return _selectOutput1_condition_xjal( this, agent );
      }
      @Override
      public void onEnter( Train agent ) {
        _selectOutput1_onEnter_xjal( this, agent );
      }
      @Override
      public void onExitFalse( Train agent ) {
        _selectOutput1_onExitFalse_xjal( this, agent );
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119625174143L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_selectOutput1_xjal( final com.anylogic.libraries.processmodeling.SelectOutput<Train> self, TableInput _t ) {
    self.conditionIsProbabilistic =
false
;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_selectOutput1_xjal( com.anylogic.libraries.processmodeling.SelectOutput<Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainCouple<Train, Train> instantiate_trainCouple2_xjal() {
    com.anylogic.libraries.rail.TrainCouple<Train, Train> _result_xjal = new com.anylogic.libraries.rail.TrainCouple<Train, Train>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainCouple2_xjal( final com.anylogic.libraries.rail.TrainCouple<Train, Train> self, TableInput _t ) {
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainCouple2_xjal( com.anylogic.libraries.rail.TrainCouple<Train, Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainMoveTo<Train> instantiate_trainMoveTo7_xjal() {
    com.anylogic.libraries.rail.TrainMoveTo<Train> _result_xjal = new com.anylogic.libraries.rail.TrainMoveTo<Train>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainMoveTo7_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, TableInput _t ) {
    self.forward = self._forward_DefaultValue_xjal();
    self.routeType = self._routeType_DefaultValue_xjal();
    self.BlockedTrackHandling = self._BlockedTrackHandling_DefaultValue_xjal();
    self.targetType = self._targetType_DefaultValue_xjal();
    self.checkFreeSpaceOnTargetTrack = self._checkFreeSpaceOnTargetTrack_DefaultValue_xjal();
    self.limitDistance = self._limitDistance_DefaultValue_xjal();
    self.distanceIs = self._distanceIs_DefaultValue_xjal();
    self.startOptions = self._startOptions_DefaultValue_xjal();
    self.finishOptions = self._finishOptions_DefaultValue_xjal();
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainMoveTo7_xjal( com.anylogic.libraries.rail.TrainMoveTo<Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainDispose<Train> instantiate_trainDispose1_xjal() {
    com.anylogic.libraries.rail.TrainDispose<Train> _result_xjal = new com.anylogic.libraries.rail.TrainDispose<Train>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainDispose1_xjal( final com.anylogic.libraries.rail.TrainDispose<Train> self, TableInput _t ) {
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainDispose1_xjal( com.anylogic.libraries.rail.TrainDispose<Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainMoveTo<Train> instantiate_trainMoveTo8_xjal() {
    com.anylogic.libraries.rail.TrainMoveTo<Train> _result_xjal = new com.anylogic.libraries.rail.TrainMoveTo<Train>( getEngine(), this, null ) {
      @Override
      public double cruiseSpeed( Train train ) {
        return _trainMoveTo8_cruiseSpeed_xjal( this, train );
      }

      @AnyLogicInternalCodegenAPI
      public SpeedUnits getUnitsForCodeOf_cruiseSpeed() {
        return MPS;
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119638866287L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_trainMoveTo8_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, TableInput _t ) {
    self.forward =
false
;
    self.routeType = self._routeType_DefaultValue_xjal();
    self.BlockedTrackHandling = self._BlockedTrackHandling_DefaultValue_xjal();
    self.targetType = self._targetType_DefaultValue_xjal();
    self.checkFreeSpaceOnTargetTrack = self._checkFreeSpaceOnTargetTrack_DefaultValue_xjal();
    self.limitDistance = self._limitDistance_DefaultValue_xjal();
    self.distanceIs = self._distanceIs_DefaultValue_xjal();
    self.startOptions = self._startOptions_DefaultValue_xjal();
    self.finishOptions = self._finishOptions_DefaultValue_xjal();
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_trainMoveTo8_xjal( com.anylogic.libraries.rail.TrainMoveTo<Train> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.rail.TrainSource<Train, Agent> instantiate_newLoco_xjal() {
    com.anylogic.libraries.rail.TrainSource<Train, Agent> _result_xjal = new com.anylogic.libraries.rail.TrainSource<Train, Agent>( getEngine(), this, null ) {
      @Override
      public int numberOfCars( Train train ) {
        return _newLoco_numberOfCars_xjal( this, train );
      }
      @Override
      public RailwayTrack track( Train train ) {
        return _newLoco_track_xjal( this, train );
      }
      @Override
      public boolean offsetFromBeginningOfTrack( Train train ) {
        return _newLoco_offsetFromBeginningOfTrack_xjal( this, train );
      }
      @Override
      public double offset( Train train, double tracklength ) {
        return _newLoco_offset_xjal( this, train, tracklength );
      }

      @AnyLogicInternalCodegenAPI
      public LengthUnits getUnitsForCodeOf_offset() {
        return METER;
      }
      @Override
      public Agent newTrain(  ) {
        return _newLoco_newTrain_xjal( this );
      }
      @Override
      public Agent newRailCar( Train train, int carindex ) {
        return _newLoco_newRailCar_xjal( this, train, carindex );
      }
      /**
       * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
       */
      @AnyLogicInternalCodegenAPI
      private static final long serialVersionUID = 2882341119638821231L;
	};

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters_newLoco_xjal( final com.anylogic.libraries.rail.TrainSource<Train, Agent> self, TableInput _t ) {
    self.arrivalType =
self.MANUAL
;
    self.firstArrivalMode = self._firstArrivalMode_DefaultValue_xjal();
    self.firstArrivalTime = self._firstArrivalTime_DefaultValue_xjal();
    self.arrivalSchedule = self._arrivalSchedule_DefaultValue_xjal();
    self.setAgentParametersFromDB = self._setAgentParametersFromDB_DefaultValue_xjal();
    self.databaseTable = self._databaseTable_DefaultValue_xjal();
    self.limitArrivals = self._limitArrivals_DefaultValue_xjal();
    self.maxArrivals = self._maxArrivals_DefaultValue_xjal();
    self.putInRailYard = self._putInRailYard_DefaultValue_xjal();
    self.locationType =
self.LOCATION_TRACK_OFFSET
;
    self.addTrainToCustomPopulation = self._addTrainToCustomPopulation_DefaultValue_xjal();
    self.addCarToCustomPopulation = self._addCarToCustomPopulation_DefaultValue_xjal();
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate_newLoco_xjal( com.anylogic.libraries.rail.TrainSource<Train, Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineN8_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineN8_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineN8;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineN8_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineN7_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineN7_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineN7;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineN7_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineN5_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineN5_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineN5;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineN5_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineN4_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineN4_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineN4;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineN4_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineN3_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineN3_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineN3;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineN3_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineN6_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineN6_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineN6;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineN6_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineN1_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineN1_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineN1;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineN1_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineN2_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineN2_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineN2;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineN2_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineN11_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineN11_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineN11;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineN11_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineN12_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineN12_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineN12;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineN12_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineN9_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineN9_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineN9;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineN9_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineN10_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineN10_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineN10;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineN10_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineHump_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineHump_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineHump;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineHump_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineArrival_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineArrival_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineArrival;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineArrival_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }
  /**
   * 创建嵌入对象实例<br>
   * <i>这个方法不应该被用户调用</i>
   */
  protected com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> instantiate__stopLineEntry_controller_xjal_xjal() {
    com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _result_xjal = new com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent>( getEngine(), this, null );

    return _result_xjal;
  }

  /**
   * 设置嵌入对象实例的参数<br>
   * 这个方法不应该被用户调用
   */
  private void setupParameters__stopLineEntry_controller_xjal_xjal( final com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> _self, TableInput _t ) {
    PositionOnTrack<Agent> self = stopLineEntry;
  }

  /**
   * 设置嵌入对象实例<br>
   * 这个方法不应该被用户调用
   */
  @AnyLogicInternalCodegenAPI
  private void doBeforeCreate__stopLineEntry_controller_xjal_xjal( com.anylogic.libraries.modules.markup_descriptors.RailStopLineDescriptor<Agent> self, TableInput _t ) {
  }

  private double _trainSource_interarrivalTime_xjal( final com.anylogic.libraries.rail.TrainSource<Train, Agent> self ) {
    double _value;
    _value =
15
;
    return _value;
  }
  private RailwayTrack _trainSource_track_xjal( final com.anylogic.libraries.rail.TrainSource<Train, Agent> self, Train train ) {
    RailwayTrack _value;
    _value =
trackEntry
;
    return _value;
  }
  private Agent _trainSource_newTrain_xjal( final com.anylogic.libraries.rail.TrainSource<Train, Agent> self ) {
    Agent _value;
    _value =
new hump_yard_4.Train()
;
    return _value;
  }
  private Agent _trainSource_newRailCar_xjal( final com.anylogic.libraries.rail.TrainSource<Train, Agent> self, Train train, int carindex ) {
    Agent _value;
    _value carindex == 0 ? new Locomotive() :
        randomlyCreate( OpenCar.class,
        BoxCar.class, GondolaCar.class,
        HopperCar.class, TankCar.class, Car006.class, Car007.class,
        Car008.class, Car009.class, Car010.class, Car011.class);
    return _value;
  }
  private PositionOnTrack<?> _trainMoveTo_positionOnTrack_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, Train train ) {
    PositionOnTrack<?> _value;
    _value =
stopLineArrival
;
    return _value;
  }
  private PositionOnTrack<?> _trainMoveTo1_positionOnTrack_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Agent> self, Agent train ) {
    PositionOnTrack<?> _value;
    _value =
stopLineHump
;
    return _value;
  }
  private RailwayTrack[] _trainMoveTo2_tracksToAvoid_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Agent> self, Agent train ) {
    RailwayTrack[] _value;
    _value = new RailwayTrack[]
{ trackArrival }
;
    return _value;
  }
  private PositionOnTrack<?> _trainMoveTo2_positionOnTrack_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Agent> self, Agent train ) {
    PositionOnTrack<?> _value;
    _value =
stopLineEntry
;
    return _value;
  }
  private PositionOnTrack<?> _trainMoveTo3_positionOnTrack_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Agent> self, Agent train ) {
    PositionOnTrack<?> _value;
    _value =
stopLineArrival
;
    return _value;
  }
  private PositionOnTrack<?> _trainMoveTo4_positionOnTrack_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, Train train ) {
    PositionOnTrack<?> _value;
    _value =
stopLineHump
;
    return _value;
  }
  private double _trainMoveTo4_cruiseSpeed_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, Train train ) {
    double _value;
    _value =
5
;
    return _value;
  }
  private double _delay_delayTime_xjal( final com.anylogic.libraries.processmodeling.Delay<Train> self, Train agent ) {
    double _value;
    _value =
15
;
    return _value;
  }
  private int _trainDecouple1_nCars_xjal( final com.anylogic.libraries.rail.TrainDecouple<Train, Train> self, Train train ) {
    int _value;
    _value =
carsOfSameType(train)
;
    return _value;
  }
  private Agent _trainDecouple1_newTrain_xjal( final com.anylogic.libraries.rail.TrainDecouple<Train, Train> self ) {
    Agent _value;
    _value =
new Train()
;
    return _value;
  }
  private double _delay1_delayTime_xjal( final com.anylogic.libraries.processmodeling.Delay<Train> self, Train agent ) {
    double _value;
    _value =
5
;
    return _value;
  }
  private boolean _selectOutput_condition_xjal( final com.anylogic.libraries.processmodeling.SelectOutput<Train> self, Train agent ) {
    boolean _value;
    _value =
agent.size()>1
;
    return _value;
  }
  private RailwayTrack _trainMoveTo5_targetTrack_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, Train train ) {
    RailwayTrack _value;
    _value =
Track12
;
    return _value;
  }
  private double _trainMoveTo5_cruiseSpeed_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, Train train ) {
    double _value;
    _value = 0;
    return _value;
  }

  private PositionOnTrack<?> _trainMoveTo6_positionOnTrack_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, Train train ) {
    PositionOnTrack<?> _value;
    _value = departurePointOnTrack(train);
    return _value;
  }

  private double _trainMoveTo6_cruiseSpeed_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, Train train ) {
    double _value;
    _value = 0;
    return _value;
  }

  private boolean _selectOutput1_condition_xjal( final com.anylogic.libraries.processmodeling.SelectOutput<Train> self, Train agent ) {
    boolean _value;
    _value = agent.size() < 8;
    return _value;
  }

  private void _selectOutput1_onEnter_xjal( final com.anylogic.libraries.processmodeling.SelectOutput<Train> self, Train agent ) {
        System.out.println("___x___>"+isOnlyOneLoco(agent));
  }
  private void _selectOutput1_onExitFalse_xjal( final com.anylogic.libraries.processmodeling.SelectOutput<Train> self, Train agent ) {
    trackReadyToDepart = agent.getTrack( true );
    System.out.println("______当前轨道 ["+trackReadyToDepart.getFullName()+"] 有几个车厢_____>"+trackReadyToDepart.getNCars());
    System.out.println("_____agent:____>"+agent.agentInfo());
    newLoco.inject();
  }
  private double _trainMoveTo8_cruiseSpeed_xjal( final com.anylogic.libraries.rail.TrainMoveTo<Train> self, Train train ) {
    double _value;
    _value =
10
;
    return _value;
  }
  private int _newLoco_numberOfCars_xjal( final com.anylogic.libraries.rail.TrainSource<Train, Agent> self, Train train ) {
    int _value;
    _value =
1
;
    return _value;
  }
  private RailwayTrack _newLoco_track_xjal( final com.anylogic.libraries.rail.TrainSource<Train, Agent> self, Train train ) {
    RailwayTrack _value;
    _value =
trackReadyToDepart
;
    return _value;
  }
  private boolean _newLoco_offsetFromBeginningOfTrack_xjal( final com.anylogic.libraries.rail.TrainSource<Train, Agent> self, Train train ) {
    boolean _value;
    _value =
true
;
    return _value;
  }
  private double _newLoco_offset_xjal( final com.anylogic.libraries.rail.TrainSource<Train, Agent> self, Train train, double tracklength ) {
    double _value;
    _value =
tracklength - 15
;
    return _value;
  }
  private Agent _newLoco_newTrain_xjal( final com.anylogic.libraries.rail.TrainSource<Train, Agent> self ) {
    Agent _value;
    _value =
new hump_yard_4.Train()
;
    return _value;
  }
  private Agent _newLoco_newRailCar_xjal( final com.anylogic.libraries.rail.TrainSource<Train, Agent> self, Train train, int carindex ) {
    Agent _value;
    _value =
new hump_yard_4.Locomotive()
;
    return _value;
  }
  // 函数


int
 carsOfSameType( Train train ) {

Agent firstCar = train.getFirst();
int n = 1;
while( train.getCar(n).getClass() == firstCar.getClass() )
	n++;
return n;
  }


PositionOnTrack<?> departurePointOnTrack( Train train ) {
    Agent car = train.getCar(0);
    if( car instanceof HopperCar ) return stopLineN1;
    if( car instanceof BoxCar ) return stopLineN2;
    if( car instanceof OpenCar ) return stopLineN3;
    if( car instanceof GondolaCar ) return stopLineN4;
    if( car instanceof TankCar ) return stopLineN5;
    if( car instanceof Car006 ) return stopLineN6;
    if( car instanceof Car007 ) return stopLineN7;
    if( car instanceof Car008 ) return stopLineN8;
    if( car instanceof Car009 ) return stopLineN9;
    if( car instanceof Car010 ) return stopLineN10;
    if( car instanceof Car011 ) return stopLineN11;
    return null;
}


boolean
 isOnlyOneLoco( Train train ) {

// 获取轨道上所有的列车
List<Agent> trainsOnTrack = train.getCars();

// 遍历列车，检查是否有名为 "newLoco" 的列车头
boolean isLocoPresent = false;
for (Agent at : trainsOnTrack) {
    if (at.getClass() == Locomotive.class) {
        isLocoPresent = true;
        break;
    }
}

// 根据结果进行操作
if (isLocoPresent) {
    System.out.println("轨道上存在名为 'newLoco' 的机车头。");
} else {
    System.out.println("轨道上不存在名为 'newLoco' 的机车头。");
}
return isLocoPresent;

  }
  // 视图区域
  public ViewArea _origin_VA = new ViewArea( this, "[原点]", 0, 0, 2040.0, 600.0 );
  @AnyLogicInternalCodegenAPI
  public ViewArea _window3d_VA = new ViewArea( this, "[window3d]", 0.0, 1200.0, 1000.0, 500.0 );
  @Override
  @AnyLogicInternalCodegenAPI
  public int getViewAreas(Map<String, ViewArea> _output) {
    if ( _output != null ) {
      _output.put( "_origin_VA", this._origin_VA );
      _output.put( "_window3d_VA", this._window3d_VA );
    }
    return 2 + super.getViewAreas( _output );
  }
  @AnyLogicInternalCodegenAPI
  protected static final int _window3d = 1;
  @AnyLogicInternalCodegenAPI
  protected static final int _Track8 = 2;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineN8 = 3;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch22 = 4;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack27 = 5;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch21 = 6;
  @AnyLogicInternalCodegenAPI
  protected static final int _Track7 = 7;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineN7 = 8;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch12 = 9;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack24 = 10;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch11 = 11;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack19 = 12;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch10 = 13;
  @AnyLogicInternalCodegenAPI
  protected static final int _trackN5 = 14;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineN5 = 15;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch7 = 16;
  @AnyLogicInternalCodegenAPI
  protected static final int _trackN4 = 17;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineN4 = 18;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch8 = 19;
  @AnyLogicInternalCodegenAPI
  protected static final int _trackN3 = 20;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineN3 = 21;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch9 = 22;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack4 = 23;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch13 = 24;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack5 = 25;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch14 = 26;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack10 = 27;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch15 = 28;
  @AnyLogicInternalCodegenAPI
  protected static final int _trackN6 = 29;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineN6 = 30;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack18 = 31;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack26 = 32;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch20 = 33;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack15 = 34;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch16 = 35;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack14 = 36;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch17 = 37;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack7 = 38;
  @AnyLogicInternalCodegenAPI
  protected static final int _trackN1 = 39;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineN1 = 40;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch18 = 41;
  @AnyLogicInternalCodegenAPI
  protected static final int _trackN2 = 42;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineN2 = 43;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack8 = 44;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch19 = 45;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack9 = 46;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack11 = 47;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack25 = 48;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch23 = 49;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack22 = 50;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch24 = 51;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack30 = 52;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch27 = 53;
  @AnyLogicInternalCodegenAPI
  protected static final int _Track11 = 54;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineN11 = 55;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch28 = 56;
  @AnyLogicInternalCodegenAPI
  protected static final int _Track12 = 57;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineN12 = 58;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack36 = 59;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch25 = 60;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack32 = 61;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch30 = 62;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack37 = 63;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch29 = 64;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack23 = 65;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack33 = 66;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch26 = 67;
  @AnyLogicInternalCodegenAPI
  protected static final int _Track9 = 68;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineN9 = 69;
  @AnyLogicInternalCodegenAPI
  protected static final int _Track10 = 70;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineN10 = 71;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack29 = 72;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack3 = 73;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineHump = 74;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch1 = 75;
  @AnyLogicInternalCodegenAPI
  protected static final int _trackArrival = 76;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineArrival = 77;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch = 78;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack = 79;
  @AnyLogicInternalCodegenAPI
  protected static final int _trackEntry = 80;
  @AnyLogicInternalCodegenAPI
  protected static final int _stopLineEntry = 81;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack1 = 82;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwaySwitch2 = 83;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack12 = 84;
  @AnyLogicInternalCodegenAPI
  protected static final int _railwayTrack2 = 85;

  /** Internal constant, shouldn't be accessed by user */
  @AnyLogicInternalCodegenAPI
  protected static final int _SHAPE_NEXT_ID_xjal = 86;

  @AnyLogicInternalCodegenAPI
  public boolean isPublicPresentationDefined() {
    return true;
  }

  @AnyLogicInternalCodegenAPI
  public boolean isEmbeddedAgentPresentationVisible( Agent _a ) {
    return super.isEmbeddedAgentPresentationVisible( _a );
  }
  @AnyLogicInternalCodegenAPI
  private void _initialize_level_xjal() {
	  level.addAll(window3d, railwayNetwork1);
  }
  @AnyLogicInternalCodegenAPI
  private void _initialize_railwayNetwork1_xjal() {
	  railwayNetwork1.addAll(Track8, stopLineN8, railwaySwitch22, railwayTrack27, railwaySwitch21, Track7, stopLineN7, railwaySwitch12, railwayTrack24, railwaySwitch11, railwayTrack19, railwaySwitch10, trackN5, stopLineN5, railwaySwitch7, trackN4, stopLineN4, railwaySwitch8, trackN3, stopLineN3, railwaySwitch9, railwayTrack4, railwaySwitch13, railwayTrack5, railwaySwitch14, railwayTrack10, railwaySwitch15, trackN6, stopLineN6, railwayTrack18, railwayTrack26, railwaySwitch20, railwayTrack15, railwaySwitch16, railwayTrack14, railwaySwitch17, railwayTrack7, trackN1, stopLineN1, railwaySwitch18, trackN2, stopLineN2, railwayTrack8, railwaySwitch19, railwayTrack9, railwayTrack11, railwayTrack25, railwaySwitch23, railwayTrack22, railwaySwitch24, railwayTrack30, railwaySwitch27, Track11, stopLineN11, railwaySwitch28, Track12, stopLineN12, railwayTrack36, railwaySwitch25, railwayTrack32, railwaySwitch30, railwayTrack37, railwaySwitch29, railwayTrack23, railwayTrack33, railwaySwitch26, Track9, stopLineN9, Track10, stopLineN10, railwayTrack29, railwayTrack3, stopLineHump, railwaySwitch1, trackArrival, stopLineArrival, railwaySwitch, railwayTrack, trackEntry, stopLineEntry, railwayTrack1, railwaySwitch2, railwayTrack12, railwayTrack2);
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _Track8_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1190.797, 282.443, 0.0, 1220.793530933433, 258.509821093446, 0.0 ),
       new MarkupSegmentLine( 1220.793530933433, 258.509821093446, 0.0, 1751.406, 255.22299999999996, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack27_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1146.333, 287.41999999999996, 0.0, 1172.792781936666, 282.5101992873263, 0.0 ),
       new MarkupSegmentLine( 1172.792781936666, 282.5101992873263, 0.0, 1190.797, 282.443, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _Track7_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1146.333, 287.41999999999996, 0.0, 1182.793045877155, 254.51009200528142, 0.0 ),
       new MarkupSegmentLine( 1182.793045877155, 254.51009200528142, 0.0, 1230.793045877155, 238.51009200528142, 0.0 ),
       new MarkupSegmentLine( 1230.793045877155, 238.51009200528142, 0.0, 1708.793045877155, 237.51009200528142, 0.0 ),
       new MarkupSegmentLine( 1708.793045877155, 237.51009200528142, 0.0, 1751.406, 255.22299999999996, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack24_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1751.406, 255.22299999999996, 0.0, 1832.254, 288.83000000000004, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack19_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1775.304, 238.928, 0.0, 1815.7935516951165, 274.50972724722345, 0.0 ),
       new MarkupSegmentLine( 1815.7935516951165, 274.50972724722345, 0.0, 1832.254, 288.83000000000004, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _trackN5_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1179.075, 198.89599999999996, 0.0, 1208.794, 189.50999999999996, 0.0 ),
       new MarkupSegmentLine( 1208.794, 189.50999999999996, 0.0, 1716.794, 187.50999999999996, 0.0 ),
       new MarkupSegmentLine( 1716.794, 187.50999999999996, 0.0, 1775.304, 238.928, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _trackN4_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1179.075, 198.89599999999996, 0.0, 1220.7936162276726, 167.50956442684438, 0.0 ),
       new MarkupSegmentLine( 1220.7936162276726, 167.50956442684438, 0.0, 1368.7936162276726, 164.50956442684438, 0.0 ),
       new MarkupSegmentLine( 1368.7936162276726, 164.50956442684438, 0.0, 1767.8649999999998, 167.52700000000004, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _trackN3_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1182.704, 158.236, 0.0, 1221.7929975550123, 145.50934963325182, 0.0 ),
       new MarkupSegmentLine( 1221.7929975550123, 145.50934963325182, 0.0, 1343.793, 143.509, 0.0 ),
       new MarkupSegmentLine( 1343.793, 143.509, 0.0, 1746.7930000000001, 145.509, 0.0 ),
       new MarkupSegmentLine( 1746.7930000000001, 145.509, 0.0, 1767.8649999999998, 167.52700000000004, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack4_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1135.793, 173.51, 0.0, 1182.704, 158.236, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack5_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1067.205, 283.05999999999995, 0.0, 1098.7930867576724, 224.510084261751, 0.0 ),
       new MarkupSegmentLine( 1098.7930867576724, 224.510084261751, 0.0, 1135.793, 173.51, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack10_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1067.205, 283.05999999999995, 0.0, 1121.222, 242.42100000000005, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _trackN6_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1121.222, 242.42100000000005, 0.0, 1214.793, 210.51000000000005, 0.0 ),
       new MarkupSegmentLine( 1214.793, 210.51000000000005, 0.0, 1715.7930000000001, 208.51000000000005, 0.0 ),
       new MarkupSegmentLine( 1715.7930000000001, 208.51000000000005, 0.0, 1775.304, 238.928, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack18_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1121.222, 242.42100000000005, 0.0, 1179.075, 198.89599999999996, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack26_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1034.275, 308.21299999999997, 0.0, 1067.205, 283.05999999999995, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack15_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 800.0, 330.0, 0.0, 1006.3931348100338, 329.5097550242042, 0.0 ),
       new MarkupSegmentArc( 1006.3931348100338, 329.5097550242042, 0.0, 1034.275, 308.21299999999997, 0.0, 1.5684210343499372, 4.060095579641224, 0.9999999999999997,
			1006.3370090758518, 305.88082093363687, 23.629000748114862, -4.714764272829649, -1.1053511194064412,
			1048.6180527817971, 326.99081971510526, 23.629000748114883, -2.678522738646298, 0.45543301110793594 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack14_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentArc( 800.0, 330.0, 0.0, 1149.593, 109.50999999999999, 0.0, 1.0081051629939721, 4.620057274082077, 0.999999999999999,
			600.1490985082819, 13.130862147062373, 374.6283936792784, -5.2750801441856146, -0.331046487423318,
			1184.1339521959833, 482.5426473274313, 374.62839367927916, -2.4645339780191398, 0.8014059449216304 ),
       new MarkupSegmentLine( 1149.593, 109.50999999999999, 0.0, 1246.793, 100.50999999999999, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack7_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1135.793, 173.51, 0.0, 1181.793, 129.51, 0.0 ),
       new MarkupSegmentLine( 1181.793, 129.51, 0.0, 1246.793, 100.50999999999999, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _trackN1_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1246.793, 100.50999999999999, 0.0, 1708.793, 101.50999999999999, 0.0 ),
       new MarkupSegmentLine( 1708.793, 101.50999999999999, 0.0, 1755.793, 122.50999999999999, 0.0 ),
       new MarkupSegmentLine( 1755.793, 122.50999999999999, 0.0, 1791.7930000000001, 155.51, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _trackN2_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1182.704, 158.236, 0.0, 1227.7929975550123, 123.50934963325182, 0.0 ),
       new MarkupSegmentLine( 1227.7929975550123, 123.50934963325182, 0.0, 1732.7929975550123, 123.50934963325182, 0.0 ),
       new MarkupSegmentLine( 1732.7929975550123, 123.50934963325182, 0.0, 1791.7930000000001, 155.51, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack8_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1791.7930000000001, 155.51, 0.0, 1820.3000000000002, 222.31500000000005, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack9_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1820.3000000000002, 222.31500000000005, 0.0, 1910.0043232266994, 309.53508856253745, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack11_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1767.8649999999998, 167.52700000000004, 0.0, 1820.3000000000002, 222.31500000000005, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack25_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1034.275, 308.21299999999997, 0.0, 1090.473, 297.78499999999997, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack22_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1090.473, 297.78499999999997, 0.0, 1145.793, 322.51, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack30_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1145.793, 322.51, 0.0, 1204.793061675145, 300.50986881366924, 0.0 ),
       new MarkupSegmentLine( 1204.793061675145, 300.50986881366924, 0.0, 1219.817, 300.602, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _Track11_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1219.817, 300.602, 0.0, 1250.7930000000001, 326.51, 0.0 ),
       new MarkupSegmentLine( 1250.7930000000001, 326.51, 0.0, 1653.7930000000001, 323.51, 0.0 ),
       new MarkupSegmentLine( 1653.7930000000001, 323.51, 0.0, 1673.7930000000001, 347.51, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _Track12_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentArc( 1145.793, 322.51, 0.0, 1249.5929999999998, 350.31, 0.0, 1.7916649211683509, -1.3258176636680323, 0.6063468130097319,
			1015.8615412475816, 901.1877566820776, 593.0852637286929, -1.3499277324214423, 0.06634174144819309,
			1640.8395078373894, -1214.6760313495583, 1613.1506774676052, -4.425178644563042, -0.04223167269478338 ),
       new MarkupSegmentLine( 1249.5929999999998, 350.31, 0.0, 1673.7930000000001, 347.51, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack36_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1673.7930000000001, 347.51, 0.0, 1753.7930000000001, 347.51, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack32_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1753.7930000000001, 347.51, 0.0, 1900.7930000000001, 347.51, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack37_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1855.473, 309.028, 0.0, 1900.7930000000001, 347.51, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack23_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1832.254, 288.83000000000004, 0.0, 1855.473, 309.028, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack33_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1718.1779999999999, 316.03200000000004, 0.0, 1730.7933317919073, 322.5101433526012, 0.0 ),
       new MarkupSegmentLine( 1730.7933317919073, 322.5101433526012, 0.0, 1753.7930000000001, 347.51, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _Track9_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1190.797, 282.443, 0.0, 1441.793266992944, 281.5099283754909, 0.0 ),
       new MarkupSegmentLine( 1441.793266992944, 281.5099283754909, 0.0, 1688.793266992944, 282.5099283754909, 0.0 ),
       new MarkupSegmentLine( 1688.793266992944, 282.5099283754909, 0.0, 1718.1779999999999, 316.03200000000004, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _Track10_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1219.817, 300.602, 0.0, 1693.7930255927738, 303.5098283778698, 0.0 ),
       new MarkupSegmentLine( 1693.7930255927738, 303.5098283778698, 0.0, 1718.1779999999999, 316.03200000000004, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack29_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1090.473, 297.78499999999997, 0.0, 1146.333, 287.41999999999996, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack3_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 670.702, 362.192, 0.0, 800.0, 330.0, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _trackArrival_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 270.149, 490.466, 0.0, 670.7016551892503, 362.1917359067553, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentArc( 270.149, 490.466, 0.0, 409.9996176442446, 479.9995346520268, 0.0, 1.6878405697605405, 4.446136931233765, 0.21926450482675724,
			313.11697858451333, 125.03505804035461, 367.94839383418014, -4.595344737419046, -0.383305210005132,
			2423.7230561455162, 7863.652142490024, 7653.326591750951, -1.837057293834384, 8.917888562084334E-6 ),
       new MarkupSegmentArc( 409.9996176442446, 479.9995346520268, 0.0, 509.9996176442446, 449.9995346520268, 0.0, 1.3045442776439713, 4.3906384259880475, 0.9514859136040759,
			819.0476856299287, 1979.8424505995345, 1554.6218493385275, -1.8370483759458223, 0.00278651835031945,
			-33.02565331341293, -1179.0762782209454, 1717.1966832562812, -4.975854511185296, -0.058285023596036545 ),
       new MarkupSegmentArc( 509.9996176442446, 449.9995346520268, 0.0, 670.7016551892503, 362.1917359067553, 0.0, 1.2490457723982544, 4.124386376837123, 2.6311740579210876,
			348.1692021091568, -35.49171195323663, 511.75270778237405, -5.034139534781332, -0.34790391483862265,
			711.7050024840701, 423.69675684898493, 73.91983556856512, -2.240450796030162, 0.0816518656876973 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _trackEntry_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 10.0, 570.0, 0.0, 270.1493823557554, 490.4664653479732, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack1_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1855.473, 309.028, 0.0, 1910.0043232266994, 309.53508856253745, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack12_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1910.0043232266994, 309.53508856253745, 0.0, 2064.527, 310.972, 0.0 ),
       new MarkupSegmentLine( 2064.527, 310.972, 0.0, 2174.527, 310.972, 0.0 ), };
  }
  @AnyLogicInternalCodegenAPI
  private static MarkupSegment[] _railwayTrack2_segments_xjal() {
    return new MarkupSegment[] {
       new MarkupSegmentLine( 1900.7930000000001, 347.51, 0.0, 2170.0, 350.0, 0.0 ), };
  }

  protected ShapeWindow3D window3d;
  protected RailwayTrack Track8;
  protected PositionOnTrack<Agent> stopLineN8;
  protected RailwaySwitch railwaySwitch22;
  protected RailwayTrack railwayTrack27;
  protected RailwaySwitch railwaySwitch21;
  protected RailwayTrack Track7;
  protected PositionOnTrack<Agent> stopLineN7;
  protected RailwaySwitch railwaySwitch12;
  protected RailwayTrack railwayTrack24;
  protected RailwaySwitch railwaySwitch11;
  protected RailwayTrack railwayTrack19;
  protected RailwaySwitch railwaySwitch10;
  protected RailwayTrack trackN5;
  protected PositionOnTrack<Agent> stopLineN5;
  protected RailwaySwitch railwaySwitch7;
  protected RailwayTrack trackN4;
  protected PositionOnTrack<Agent> stopLineN4;
  protected RailwaySwitch railwaySwitch8;
  protected RailwayTrack trackN3;
  protected PositionOnTrack<Agent> stopLineN3;
  protected RailwaySwitch railwaySwitch9;
  protected RailwayTrack railwayTrack4;
  protected RailwaySwitch railwaySwitch13;
  protected RailwayTrack railwayTrack5;
  protected RailwaySwitch railwaySwitch14;
  protected RailwayTrack railwayTrack10;
  protected RailwaySwitch railwaySwitch15;
  protected RailwayTrack trackN6;
  protected PositionOnTrack<Agent> stopLineN6;
  protected RailwayTrack railwayTrack18;
  protected RailwayTrack railwayTrack26;
  protected RailwaySwitch railwaySwitch20;
  protected RailwayTrack railwayTrack15;
  protected RailwaySwitch railwaySwitch16;
  protected RailwayTrack railwayTrack14;
  protected RailwaySwitch railwaySwitch17;
  protected RailwayTrack railwayTrack7;
  protected RailwayTrack trackN1;
  protected PositionOnTrack<Agent> stopLineN1;
  protected RailwaySwitch railwaySwitch18;
  protected RailwayTrack trackN2;
  protected PositionOnTrack<Agent> stopLineN2;
  protected RailwayTrack railwayTrack8;
  protected RailwaySwitch railwaySwitch19;
  protected RailwayTrack railwayTrack9;
  protected RailwayTrack railwayTrack11;
  protected RailwayTrack railwayTrack25;
  protected RailwaySwitch railwaySwitch23;
  protected RailwayTrack railwayTrack22;
  protected RailwaySwitch railwaySwitch24;
  protected RailwayTrack railwayTrack30;
  protected RailwaySwitch railwaySwitch27;
  protected RailwayTrack Track11;
  protected PositionOnTrack<Agent> stopLineN11;
  protected RailwaySwitch railwaySwitch28;
  protected RailwayTrack Track12;
  protected PositionOnTrack<Agent> stopLineN12;
  protected RailwayTrack railwayTrack36;
  protected RailwaySwitch railwaySwitch25;
  protected RailwayTrack railwayTrack32;
  protected RailwaySwitch railwaySwitch30;
  protected RailwayTrack railwayTrack37;
  protected RailwaySwitch railwaySwitch29;
  protected RailwayTrack railwayTrack23;
  protected RailwayTrack railwayTrack33;
  protected RailwaySwitch railwaySwitch26;
  protected RailwayTrack Track9;
  protected PositionOnTrack<Agent> stopLineN9;
  protected RailwayTrack Track10;
  protected PositionOnTrack<Agent> stopLineN10;
  protected RailwayTrack railwayTrack29;
  protected RailwayTrack railwayTrack3;
  protected PositionOnTrack<Agent> stopLineHump;
  protected RailwaySwitch railwaySwitch1;
  protected RailwayTrack trackArrival;
  protected PositionOnTrack<Agent> stopLineArrival;
  protected RailwaySwitch railwaySwitch;
  protected RailwayTrack railwayTrack;
  protected RailwayTrack trackEntry;
  protected PositionOnTrack<Agent> stopLineEntry;
  protected RailwayTrack railwayTrack1;
  protected RailwaySwitch railwaySwitch2;
  protected RailwayTrack railwayTrack12;
  protected RailwayTrack railwayTrack2;
  protected com.anylogic.engine.markup.Level level;

  private com.anylogic.engine.markup.Level[] _getLevels_xjal;

  @Override
  public com.anylogic.engine.markup.Level[] getLevels() {
    return _getLevels_xjal;
  }

  protected com.anylogic.engine.markup.RailwayNetwork railwayNetwork1;

  private com.anylogic.engine.markup.RailwayNetwork[] _getRailwayNetworks_xjal;

  @Override
  public com.anylogic.engine.markup.RailwayNetwork[] getRailwayNetworks() {
    return _getRailwayNetworks_xjal;
  }

  @AnyLogicInternalCodegenAPI
  private void _createPersistentElementsBP0_xjal() {
    window3d = new ShapeWindow3D( Main.this, false, 0.0, 1200.0, 1000.0, 500.0, WINDOW_3D_NAVIGATION_FULL, 2000.0 );

    Track8 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _Track8_segments_xjal() );

    railwayTrack27 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack27_segments_xjal() );

    Track7 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _Track7_segments_xjal() );

    railwayTrack24 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack24_segments_xjal() );

    railwayTrack19 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack19_segments_xjal() );

    trackN5 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _trackN5_segments_xjal() );

    trackN4 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _trackN4_segments_xjal() );

    trackN3 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _trackN3_segments_xjal() );

    railwayTrack4 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack4_segments_xjal() );

    railwayTrack5 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack5_segments_xjal() );

    railwayTrack10 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack10_segments_xjal() );

    trackN6 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _trackN6_segments_xjal() );

    railwayTrack18 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack18_segments_xjal() );

    railwayTrack26 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack26_segments_xjal() );

    railwayTrack15 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack15_segments_xjal() );

    railwayTrack14 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack14_segments_xjal() );

    railwayTrack7 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack7_segments_xjal() );

    trackN1 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _trackN1_segments_xjal() );

    trackN2 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _trackN2_segments_xjal() );

    railwayTrack8 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack8_segments_xjal() );

    railwayTrack9 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack9_segments_xjal() );

    railwayTrack11 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack11_segments_xjal() );

    railwayTrack25 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack25_segments_xjal() );

    railwayTrack22 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack22_segments_xjal() );

    railwayTrack30 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack30_segments_xjal() );

    Track11 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _Track11_segments_xjal() );

    Track12 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _Track12_segments_xjal() );

    railwayTrack36 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack36_segments_xjal() );

    railwayTrack32 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack32_segments_xjal() );

    railwayTrack37 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack37_segments_xjal() );

    railwayTrack23 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack23_segments_xjal() );

    railwayTrack33 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack33_segments_xjal() );

    Track9 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _Track9_segments_xjal() );

    Track10 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _Track10_segments_xjal() );

    railwayTrack29 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack29_segments_xjal() );

    railwayTrack3 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack3_segments_xjal() );

    trackArrival = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 2.0, _trackArrival_segments_xjal() );

    railwayTrack = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 2.0, _railwayTrack_segments_xjal() );

    trackEntry = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 2.0, _trackEntry_segments_xjal() );

    railwayTrack1 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack1_segments_xjal() );

    railwayTrack12 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack12_segments_xjal() );

    railwayTrack2 = new RailwayTrack( this, SHAPE_DRAW_2D3D, true, PATH_RAILROAD, black, 1.5, _railwayTrack2_segments_xjal() );

    railwaySwitch22 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, Track8, railwayTrack27, Track9 );

    railwaySwitch21 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack27, Track7, railwayTrack29 );

    railwaySwitch12 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, Track8, Track7, railwayTrack24 );

    railwaySwitch11 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack24, railwayTrack19, railwayTrack23 );

    railwaySwitch10 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack19, trackN5, trackN6 );

    railwaySwitch7 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, trackN5, trackN4, railwayTrack18 );

    railwaySwitch8 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, trackN4, trackN3, railwayTrack11 );

    railwaySwitch9 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, trackN3, railwayTrack4, trackN2 );

    railwaySwitch13 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack4, railwayTrack5, railwayTrack7 );

    railwaySwitch14 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack5, railwayTrack10, railwayTrack26 );

    railwaySwitch15 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack10, trackN6, railwayTrack18 );

    railwaySwitch20 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack26, railwayTrack15, railwayTrack25 );

    railwaySwitch16 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack15, railwayTrack14, railwayTrack3 );

    railwaySwitch17 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack14, railwayTrack7, trackN1 );

    railwaySwitch18 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, trackN1, trackN2, railwayTrack8 );

    railwaySwitch19 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack8, railwayTrack9, railwayTrack11 );

    railwaySwitch23 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack25, railwayTrack22, railwayTrack29 );

    railwaySwitch24 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack22, railwayTrack30, Track12 );

    railwaySwitch27 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack30, Track11, Track10 );

    railwaySwitch28 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, Track11, Track12, railwayTrack36 );

    railwaySwitch25 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack36, railwayTrack32, railwayTrack33 );

    railwaySwitch30 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack32, railwayTrack37, railwayTrack2 );

    railwaySwitch29 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack37, railwayTrack23, railwayTrack1 );

    railwaySwitch26 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack33, Track9, Track10 );

    railwaySwitch1 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack3, trackArrival, railwayTrack );

    railwaySwitch = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, trackArrival, railwayTrack, trackEntry );

    railwaySwitch2 = new RailwaySwitch( this, SHAPE_DRAW_2D3D, true, 2.25, black, gray,
			RailwaySwitch.IRailwaySwitchType.createAllToAll()
		, railwayTrack9, railwayTrack1, railwayTrack12 );

  }

  @AnyLogicInternalCodegenAPI
  private void _createPersistentElementsAP0_xjal() {
    {
    stopLineN8 = new PositionOnTrack<Agent>( this, _stopLineN8_controller_xjal, SHAPE_DRAW_2D3D, true, black, Track8, 350.0 );

    }
    {
    stopLineN7 = new PositionOnTrack<Agent>( this, _stopLineN7_controller_xjal, SHAPE_DRAW_2D3D, true, black, Track7, 400.0 );

    }
    {
    stopLineN5 = new PositionOnTrack<Agent>( this, _stopLineN5_controller_xjal, SHAPE_DRAW_2D3D, true, black, trackN5, 350.0 );

    }
    {
    stopLineN4 = new PositionOnTrack<Agent>( this, _stopLineN4_controller_xjal, SHAPE_DRAW_2D3D, true, black, trackN4, 360.0 );

    }
    {
    stopLineN3 = new PositionOnTrack<Agent>( this, _stopLineN3_controller_xjal, SHAPE_DRAW_2D3D, true, black, trackN3, 350.0 );

    }
    {
    stopLineN6 = new PositionOnTrack<Agent>( this, _stopLineN6_controller_xjal, SHAPE_DRAW_2D3D, true, black, trackN6, 410.0 );

    }
    {
    stopLineN1 = new PositionOnTrack<Agent>( this, _stopLineN1_controller_xjal, SHAPE_DRAW_2D3D, true, black, trackN1, 280.0 );

    }
    {
    stopLineN2 = new PositionOnTrack<Agent>( this, _stopLineN2_controller_xjal, SHAPE_DRAW_2D3D, true, black, trackN2, 360.0 );

    }
    {
    stopLineN11 = new PositionOnTrack<Agent>( this, _stopLineN11_controller_xjal, SHAPE_DRAW_2D3D, true, black, Track11, 320.0 );

    }
    {
    stopLineN12 = new PositionOnTrack<Agent>( this, _stopLineN12_controller_xjal, SHAPE_DRAW_2D3D, true, black, Track12, 390.0 );

    }
    {
    stopLineN9 = new PositionOnTrack<Agent>( this, _stopLineN9_controller_xjal, SHAPE_DRAW_2D3D, true, black, Track9, 340.0 );

    }
    {
    stopLineN10 = new PositionOnTrack<Agent>( this, _stopLineN10_controller_xjal, SHAPE_DRAW_2D3D, true, black, Track10, 310.0 );

    }
    {
    stopLineHump = new PositionOnTrack<Agent>( this, _stopLineHump_controller_xjal, SHAPE_DRAW_2D3D, true, black, railwayTrack3, 80.0 );

    }
    {
    stopLineArrival = new PositionOnTrack<Agent>( this, _stopLineArrival_controller_xjal, SHAPE_DRAW_2D3D, true, black, trackArrival, 388.9703484108029 );

    }
    {
    stopLineEntry = new PositionOnTrack<Agent>( this, _stopLineEntry_controller_xjal, SHAPE_DRAW_2D3D, true, black, trackEntry, 221.28997689564545 );

    }
  }

  @AnyLogicInternalCodegenAPI
  private void _createPersistentElementsBS0_xjal() {
  }


  // 持久元素的静态初始化
  private void instantiatePersistentElements_xjal() {
    level = new com.anylogic.engine.markup.Level(this, "level", SHAPE_DRAW_2D3D, 0.0, true, true);
	_getLevels_xjal = new com.anylogic.engine.markup.Level[] {
      level };
    railwayNetwork1 = new com.anylogic.engine.markup.RailwayNetwork(this, "railwayNetwork1", SHAPE_DRAW_2D3D, 0.0, true, true);
	_getRailwayNetworks_xjal = new com.anylogic.engine.markup.RailwayNetwork[] {
      railwayNetwork1 };
    _createPersistentElementsBP0_xjal();
  }
  protected ShapeTopLevelPresentationGroup presentation;
  protected ShapeModelElementsGroup icon;

  @Override
  @AnyLogicInternalCodegenAPI
  public ShapeTopLevelPresentationGroup getPresentationShape() {
    return presentation;
  }

  @Override
  @AnyLogicInternalCodegenAPI
  public ShapeModelElementsGroup getModelElementsShape() {
    return icon;
  }




  /**
   * 构造器
   */
  public Main( Engine engine, Agent owner, AgentList<? extends Main> ownerPopulation ) {
    super( engine, owner, ownerPopulation );
    instantiateBaseStructureThis_xjal();
  }

  @AnyLogicInternalCodegenAPI
  public void onOwnerChanged_xjal() {
    super.onOwnerChanged_xjal();
    setupReferences_xjal();
  }

  @AnyLogicInternalCodegenAPI
  public void instantiateBaseStructure_xjal() {
    super.instantiateBaseStructure_xjal();
    instantiateBaseStructureThis_xjal();
  }

  @AnyLogicInternalCodegenAPI
  private void instantiateBaseStructureThis_xjal() {
    trainSource = instantiate_trainSource_xjal();
    trainMoveTo = instantiate_trainMoveTo_xjal();
    trainDecouple = instantiate_trainDecouple_xjal();
    trainCouple = instantiate_trainCouple_xjal();
    trainDispose = instantiate_trainDispose_xjal();
    trainMoveTo1 = instantiate_trainMoveTo1_xjal();
    trainMoveTo2 = instantiate_trainMoveTo2_xjal();
    trainMoveTo3 = instantiate_trainMoveTo3_xjal();
    trainMoveTo4 = instantiate_trainMoveTo4_xjal();
    delay = instantiate_delay_xjal();
    trainDecouple1 = instantiate_trainDecouple1_xjal();
    delay1 = instantiate_delay1_xjal();
    selectOutput = instantiate_selectOutput_xjal();
    trainMoveTo5 = instantiate_trainMoveTo5_xjal();
    trainMoveTo6 = instantiate_trainMoveTo6_xjal();
    trainCouple1 = instantiate_trainCouple1_xjal();
    selectOutput1 = instantiate_selectOutput1_xjal();
    trainCouple2 = instantiate_trainCouple2_xjal();
    trainMoveTo7 = instantiate_trainMoveTo7_xjal();
    trainDispose1 = instantiate_trainDispose1_xjal();
    trainMoveTo8 = instantiate_trainMoveTo8_xjal();
    newLoco = instantiate_newLoco_xjal();
    _stopLineN8_controller_xjal = instantiate__stopLineN8_controller_xjal_xjal();
    _stopLineN7_controller_xjal = instantiate__stopLineN7_controller_xjal_xjal();
    _stopLineN5_controller_xjal = instantiate__stopLineN5_controller_xjal_xjal();
    _stopLineN4_controller_xjal = instantiate__stopLineN4_controller_xjal_xjal();
    _stopLineN3_controller_xjal = instantiate__stopLineN3_controller_xjal_xjal();
    _stopLineN6_controller_xjal = instantiate__stopLineN6_controller_xjal_xjal();
    _stopLineN1_controller_xjal = instantiate__stopLineN1_controller_xjal_xjal();
    _stopLineN2_controller_xjal = instantiate__stopLineN2_controller_xjal_xjal();
    _stopLineN11_controller_xjal = instantiate__stopLineN11_controller_xjal_xjal();
    _stopLineN12_controller_xjal = instantiate__stopLineN12_controller_xjal_xjal();
    _stopLineN9_controller_xjal = instantiate__stopLineN9_controller_xjal_xjal();
    _stopLineN10_controller_xjal = instantiate__stopLineN10_controller_xjal_xjal();
    _stopLineHump_controller_xjal = instantiate__stopLineHump_controller_xjal_xjal();
    _stopLineArrival_controller_xjal = instantiate__stopLineArrival_controller_xjal_xjal();
    _stopLineEntry_controller_xjal = instantiate__stopLineEntry_controller_xjal_xjal();
	instantiatePersistentElements_xjal();
    setupReferences_xjal();
  }

  @AnyLogicInternalCodegenAPI
  private void setupReferences_xjal() {
  }

  /**
   * Simple constructor. Please add created agent to some population by calling goToPopulation() function
   */
  public Main() {
  }

  /**
   * 创建嵌入对象实例
   */
  @AnyLogicInternalCodegenAPI
  private void instantiatePopulations_xjal() {
  }

  @Override
  @AnyLogicInternalCodegenAPI
  public void doCreate() {
    super.doCreate();
    // 创建嵌入对象实例
    instantiatePopulations_xjal();
    // 为简单变量赋初始值
    setupPlainVariables_Main_xjal();
    // 持久元素的动态初始化
    _createPersistentElementsAP0_xjal();
	_initialize_level_xjal();
	_initialize_railwayNetwork1_xjal();
    level.initialize();
    presentation = new ShapeTopLevelPresentationGroup( Main.this, true, 0, 0, 0, 0 , level,
      // default light
      new Light3D.Daylight( null, false, true ) );
    window3d.setCamera( new Camera3D( 1374, 576, 300, 0.7853981633974483, -1.5707963267948966 ), false );
    // 创建嵌入对象实例
    instantiatePopulations_xjal();
    icon = new ShapeModelElementsGroup( Main.this, getElementProperty( "hump_yard_4.Main.icon", IElementDescriptor.MODEL_ELEMENT_DESCRIPTORS )  );
    icon.setIconOffsets( 0.0, 0.0 );
    // 创建非重复嵌入对象
    setupParameters_trainSource_xjal( trainSource, null );
    doBeforeCreate_trainSource_xjal( trainSource, null );
    trainSource.createAsEmbedded();
    setupParameters_trainMoveTo_xjal( trainMoveTo, null );
    doBeforeCreate_trainMoveTo_xjal( trainMoveTo, null );
    trainMoveTo.createAsEmbedded();
    setupParameters_trainDecouple_xjal( trainDecouple, null );
    doBeforeCreate_trainDecouple_xjal( trainDecouple, null );
    trainDecouple.createAsEmbedded();
    setupParameters_trainCouple_xjal( trainCouple, null );
    doBeforeCreate_trainCouple_xjal( trainCouple, null );
    trainCouple.createAsEmbedded();
    setupParameters_trainDispose_xjal( trainDispose, null );
    doBeforeCreate_trainDispose_xjal( trainDispose, null );
    trainDispose.createAsEmbedded();
    setupParameters_trainMoveTo1_xjal( trainMoveTo1, null );
    doBeforeCreate_trainMoveTo1_xjal( trainMoveTo1, null );
    trainMoveTo1.createAsEmbedded();
    setupParameters_trainMoveTo2_xjal( trainMoveTo2, null );
    doBeforeCreate_trainMoveTo2_xjal( trainMoveTo2, null );
    trainMoveTo2.createAsEmbedded();
    setupParameters_trainMoveTo3_xjal( trainMoveTo3, null );
    doBeforeCreate_trainMoveTo3_xjal( trainMoveTo3, null );
    trainMoveTo3.createAsEmbedded();
    setupParameters_trainMoveTo4_xjal( trainMoveTo4, null );
    doBeforeCreate_trainMoveTo4_xjal( trainMoveTo4, null );
    trainMoveTo4.createAsEmbedded();
    setupParameters_delay_xjal( delay, null );
    doBeforeCreate_delay_xjal( delay, null );
    delay.createAsEmbedded();
    setupParameters_trainDecouple1_xjal( trainDecouple1, null );
    doBeforeCreate_trainDecouple1_xjal( trainDecouple1, null );
    trainDecouple1.createAsEmbedded();
    setupParameters_delay1_xjal( delay1, null );
    doBeforeCreate_delay1_xjal( delay1, null );
    delay1.createAsEmbedded();
    setupParameters_selectOutput_xjal( selectOutput, null );
    doBeforeCreate_selectOutput_xjal( selectOutput, null );
    selectOutput.createAsEmbedded();
    setupParameters_trainMoveTo5_xjal( trainMoveTo5, null );
    doBeforeCreate_trainMoveTo5_xjal( trainMoveTo5, null );
    trainMoveTo5.createAsEmbedded();
    setupParameters_trainMoveTo6_xjal( trainMoveTo6, null );
    doBeforeCreate_trainMoveTo6_xjal( trainMoveTo6, null );
    trainMoveTo6.createAsEmbedded();
    setupParameters_trainCouple1_xjal( trainCouple1, null );
    doBeforeCreate_trainCouple1_xjal( trainCouple1, null );
    trainCouple1.createAsEmbedded();
    setupParameters_selectOutput1_xjal( selectOutput1, null );
    doBeforeCreate_selectOutput1_xjal( selectOutput1, null );
    selectOutput1.createAsEmbedded();
    setupParameters_trainCouple2_xjal( trainCouple2, null );
    doBeforeCreate_trainCouple2_xjal( trainCouple2, null );
    trainCouple2.createAsEmbedded();
    setupParameters_trainMoveTo7_xjal( trainMoveTo7, null );
    doBeforeCreate_trainMoveTo7_xjal( trainMoveTo7, null );
    trainMoveTo7.createAsEmbedded();
    setupParameters_trainDispose1_xjal( trainDispose1, null );
    doBeforeCreate_trainDispose1_xjal( trainDispose1, null );
    trainDispose1.createAsEmbedded();
    setupParameters_trainMoveTo8_xjal( trainMoveTo8, null );
    doBeforeCreate_trainMoveTo8_xjal( trainMoveTo8, null );
    trainMoveTo8.createAsEmbedded();
    setupParameters_newLoco_xjal( newLoco, null );
    doBeforeCreate_newLoco_xjal( newLoco, null );
    newLoco.createAsEmbedded();
    setupParameters__stopLineN8_controller_xjal_xjal( _stopLineN8_controller_xjal, null );
    doBeforeCreate__stopLineN8_controller_xjal_xjal( _stopLineN8_controller_xjal, null );
    _stopLineN8_controller_xjal.createAsEmbedded();
    setupParameters__stopLineN7_controller_xjal_xjal( _stopLineN7_controller_xjal, null );
    doBeforeCreate__stopLineN7_controller_xjal_xjal( _stopLineN7_controller_xjal, null );
    _stopLineN7_controller_xjal.createAsEmbedded();
    setupParameters__stopLineN5_controller_xjal_xjal( _stopLineN5_controller_xjal, null );
    doBeforeCreate__stopLineN5_controller_xjal_xjal( _stopLineN5_controller_xjal, null );
    _stopLineN5_controller_xjal.createAsEmbedded();
    setupParameters__stopLineN4_controller_xjal_xjal( _stopLineN4_controller_xjal, null );
    doBeforeCreate__stopLineN4_controller_xjal_xjal( _stopLineN4_controller_xjal, null );
    _stopLineN4_controller_xjal.createAsEmbedded();
    setupParameters__stopLineN3_controller_xjal_xjal( _stopLineN3_controller_xjal, null );
    doBeforeCreate__stopLineN3_controller_xjal_xjal( _stopLineN3_controller_xjal, null );
    _stopLineN3_controller_xjal.createAsEmbedded();
    setupParameters__stopLineN6_controller_xjal_xjal( _stopLineN6_controller_xjal, null );
    doBeforeCreate__stopLineN6_controller_xjal_xjal( _stopLineN6_controller_xjal, null );
    _stopLineN6_controller_xjal.createAsEmbedded();
    setupParameters__stopLineN1_controller_xjal_xjal( _stopLineN1_controller_xjal, null );
    doBeforeCreate__stopLineN1_controller_xjal_xjal( _stopLineN1_controller_xjal, null );
    _stopLineN1_controller_xjal.createAsEmbedded();
    setupParameters__stopLineN2_controller_xjal_xjal( _stopLineN2_controller_xjal, null );
    doBeforeCreate__stopLineN2_controller_xjal_xjal( _stopLineN2_controller_xjal, null );
    _stopLineN2_controller_xjal.createAsEmbedded();
    setupParameters__stopLineN11_controller_xjal_xjal( _stopLineN11_controller_xjal, null );
    doBeforeCreate__stopLineN11_controller_xjal_xjal( _stopLineN11_controller_xjal, null );
    _stopLineN11_controller_xjal.createAsEmbedded();
    setupParameters__stopLineN12_controller_xjal_xjal( _stopLineN12_controller_xjal, null );
    doBeforeCreate__stopLineN12_controller_xjal_xjal( _stopLineN12_controller_xjal, null );
    _stopLineN12_controller_xjal.createAsEmbedded();
    setupParameters__stopLineN9_controller_xjal_xjal( _stopLineN9_controller_xjal, null );
    doBeforeCreate__stopLineN9_controller_xjal_xjal( _stopLineN9_controller_xjal, null );
    _stopLineN9_controller_xjal.createAsEmbedded();
    setupParameters__stopLineN10_controller_xjal_xjal( _stopLineN10_controller_xjal, null );
    doBeforeCreate__stopLineN10_controller_xjal_xjal( _stopLineN10_controller_xjal, null );
    _stopLineN10_controller_xjal.createAsEmbedded();
    setupParameters__stopLineHump_controller_xjal_xjal( _stopLineHump_controller_xjal, null );
    doBeforeCreate__stopLineHump_controller_xjal_xjal( _stopLineHump_controller_xjal, null );
    _stopLineHump_controller_xjal.createAsEmbedded();
    setupParameters__stopLineArrival_controller_xjal_xjal( _stopLineArrival_controller_xjal, null );
    doBeforeCreate__stopLineArrival_controller_xjal_xjal( _stopLineArrival_controller_xjal, null );
    _stopLineArrival_controller_xjal.createAsEmbedded();
    setupParameters__stopLineEntry_controller_xjal_xjal( _stopLineEntry_controller_xjal, null );
    doBeforeCreate__stopLineEntry_controller_xjal_xjal( _stopLineEntry_controller_xjal, null );
    _stopLineEntry_controller_xjal.createAsEmbedded();
	 // 非重复对象的端口连接器
    trainMoveTo.in.connect( trainSource.out ); // connector
    trainDecouple.in.connect( trainMoveTo.out ); // connector1
    trainDecouple.out.connect( trainCouple.in1 ); // connector2
    trainMoveTo1.in.connect( trainDecouple.outDecoupled ); // connector4
    trainMoveTo2.in.connect( trainMoveTo1.out ); // connector5
    trainMoveTo3.in.connect( trainMoveTo2.out ); // connector6
    trainCouple.out.connect( trainMoveTo4.in ); // connector3
    delay.in.connect( trainMoveTo4.out ); // connector9
    trainDecouple1.in.connect( delay.out ); // connector10
    trainMoveTo3.out.connect( trainCouple.in2 ); // connector7
    trainCouple.in2.connect( trainMoveTo3.outHit ); // connector8
    trainDecouple1.in.connect( delay.out ); // connector11
    delay1.in.connect( trainDecouple1.out ); // connector12
    selectOutput.in.connect( delay1.out ); // connector13
    selectOutput.outT.connect( trainMoveTo4.in ); // connector14
    trainMoveTo5.in.connect( selectOutput.outF ); // connector15
    trainDispose.in.connect( trainMoveTo5.out ); // connector16
    trainMoveTo6.in.connect( trainDecouple1.outDecoupled ); // connector17
    trainCouple1.in1.connect( trainMoveTo6.out ); // connector18
    selectOutput1.in.connect( trainCouple1.out ); // connector19
    selectOutput1.outT.connect( trainCouple1.in1 ); // connector20
    trainCouple1.in2.connect( trainMoveTo6.outHit ); // connector21
    trainCouple2.in1.connect( selectOutput1.outF ); // connector22
    trainMoveTo7.in.connect( trainCouple2.out ); // connector23
    trainDispose1.in.connect( trainMoveTo7.out ); // connector24
    trainMoveTo8.out.connect( trainCouple2.in2 ); // connector25
    trainMoveTo8.outHit.connect( trainCouple2.in2 ); // connector26
    newLoco.out.connect( trainMoveTo8.in ); // connector27
    // 创建重复嵌入对象
    setupInitialConditions_xjal( Main.class );
    // 持久元素的动态初始化
    _createPersistentElementsBS0_xjal();
  }

  @Override
  @AnyLogicInternalCodegenAPI
  public void doStart() {
    super.doStart();
    trainSource.startAsEmbedded();
    trainMoveTo.startAsEmbedded();
    trainDecouple.startAsEmbedded();
    trainCouple.startAsEmbedded();
    trainDispose.startAsEmbedded();
    trainMoveTo1.startAsEmbedded();
    trainMoveTo2.startAsEmbedded();
    trainMoveTo3.startAsEmbedded();
    trainMoveTo4.startAsEmbedded();
    delay.startAsEmbedded();
    trainDecouple1.startAsEmbedded();
    delay1.startAsEmbedded();
    selectOutput.startAsEmbedded();
    trainMoveTo5.startAsEmbedded();
    trainMoveTo6.startAsEmbedded();
    trainCouple1.startAsEmbedded();
    selectOutput1.startAsEmbedded();
    trainCouple2.startAsEmbedded();
    trainMoveTo7.startAsEmbedded();
    trainDispose1.startAsEmbedded();
    trainMoveTo8.startAsEmbedded();
    newLoco.startAsEmbedded();
    _stopLineN8_controller_xjal.startAsEmbedded();
    _stopLineN7_controller_xjal.startAsEmbedded();
    _stopLineN5_controller_xjal.startAsEmbedded();
    _stopLineN4_controller_xjal.startAsEmbedded();
    _stopLineN3_controller_xjal.startAsEmbedded();
    _stopLineN6_controller_xjal.startAsEmbedded();
    _stopLineN1_controller_xjal.startAsEmbedded();
    _stopLineN2_controller_xjal.startAsEmbedded();
    _stopLineN11_controller_xjal.startAsEmbedded();
    _stopLineN12_controller_xjal.startAsEmbedded();
    _stopLineN9_controller_xjal.startAsEmbedded();
    _stopLineN10_controller_xjal.startAsEmbedded();
    _stopLineHump_controller_xjal.startAsEmbedded();
    _stopLineArrival_controller_xjal.startAsEmbedded();
    _stopLineEntry_controller_xjal.startAsEmbedded();
  }


  /**
   * 为简单变量赋初始值<br>
   * <em>This method isn't designed to be called by user and may be removed in future releases.</em>
   */
  @AnyLogicInternalCodegenAPI
  public void setupPlainVariables_xjal() {
    setupPlainVariables_Main_xjal();
  }

  /**
   * 为简单变量赋初始值<br>
   * <em>This method isn't designed to be called by user and may be removed in future releases.</em>
   */
  @AnyLogicInternalCodegenAPI
  private void setupPlainVariables_Main_xjal() {
  }

  // 用户API -----------------------------------------------------
  @AnyLogicInternalCodegenAPI
  public static LinkToAgentAnimationSettings _connections_commonAnimationSettings_xjal = new LinkToAgentAnimationSettingsImpl( false, black, 1.0, LINE_STYLE_SOLID, ARROW_NONE, 0.0 );

  public LinkToAgentCollection<Agent, Agent> connections = new LinkToAgentStandardImpl<Agent, Agent>(this, _connections_commonAnimationSettings_xjal);
  @Override
  public LinkToAgentCollection<? extends Agent, ? extends Agent> getLinkToAgentStandard_xjal() {
    return connections;
  }


  @AnyLogicInternalCodegenAPI
  public void drawLinksToAgents(boolean _underAgents_xjal, LinkToAgentAnimator _animator_xjal) {
    super.drawLinksToAgents(_underAgents_xjal, _animator_xjal);
    if ( _underAgents_xjal ) {
      _animator_xjal.drawLink( this, connections, true, true );
    }
  }

  public List<Object> getEmbeddedObjects() {
    List<Object> list = super.getEmbeddedObjects();
    if (list == null) {
      list = new LinkedList<Object>();
    }
    list.add( trainSource );
    list.add( trainMoveTo );
    list.add( trainDecouple );
    list.add( trainCouple );
    list.add( trainDispose );
    list.add( trainMoveTo1 );
    list.add( trainMoveTo2 );
    list.add( trainMoveTo3 );
    list.add( trainMoveTo4 );
    list.add( delay );
    list.add( trainDecouple1 );
    list.add( delay1 );
    list.add( selectOutput );
    list.add( trainMoveTo5 );
    list.add( trainMoveTo6 );
    list.add( trainCouple1 );
    list.add( selectOutput1 );
    list.add( trainCouple2 );
    list.add( trainMoveTo7 );
    list.add( trainDispose1 );
    list.add( trainMoveTo8 );
    list.add( newLoco );
    list.add( _stopLineN8_controller_xjal );
    list.add( _stopLineN7_controller_xjal );
    list.add( _stopLineN5_controller_xjal );
    list.add( _stopLineN4_controller_xjal );
    list.add( _stopLineN3_controller_xjal );
    list.add( _stopLineN6_controller_xjal );
    list.add( _stopLineN1_controller_xjal );
    list.add( _stopLineN2_controller_xjal );
    list.add( _stopLineN11_controller_xjal );
    list.add( _stopLineN12_controller_xjal );
    list.add( _stopLineN9_controller_xjal );
    list.add( _stopLineN10_controller_xjal );
    list.add( _stopLineHump_controller_xjal );
    list.add( _stopLineArrival_controller_xjal );
    list.add( _stopLineEntry_controller_xjal );
    return list;
  }

  public AgentList<? extends Main> getPopulation() {
    return (AgentList<? extends Main>) super.getPopulation();
  }

  public List<? extends Main> agentsInRange( double distance ) {
    return (List<? extends Main>) super.agentsInRange( distance );
  }

  @AnyLogicInternalCodegenAPI
  public void onDestroy() {
    trainSource.onDestroy();
    trainMoveTo.onDestroy();
    trainDecouple.onDestroy();
    trainCouple.onDestroy();
    trainDispose.onDestroy();
    trainMoveTo1.onDestroy();
    trainMoveTo2.onDestroy();
    trainMoveTo3.onDestroy();
    trainMoveTo4.onDestroy();
    delay.onDestroy();
    trainDecouple1.onDestroy();
    delay1.onDestroy();
    selectOutput.onDestroy();
    trainMoveTo5.onDestroy();
    trainMoveTo6.onDestroy();
    trainCouple1.onDestroy();
    selectOutput1.onDestroy();
    trainCouple2.onDestroy();
    trainMoveTo7.onDestroy();
    trainDispose1.onDestroy();
    trainMoveTo8.onDestroy();
    newLoco.onDestroy();
    _stopLineN8_controller_xjal.onDestroy();
    _stopLineN7_controller_xjal.onDestroy();
    _stopLineN5_controller_xjal.onDestroy();
    _stopLineN4_controller_xjal.onDestroy();
    _stopLineN3_controller_xjal.onDestroy();
    _stopLineN6_controller_xjal.onDestroy();
    _stopLineN1_controller_xjal.onDestroy();
    _stopLineN2_controller_xjal.onDestroy();
    _stopLineN11_controller_xjal.onDestroy();
    _stopLineN12_controller_xjal.onDestroy();
    _stopLineN9_controller_xjal.onDestroy();
    _stopLineN10_controller_xjal.onDestroy();
    _stopLineHump_controller_xjal.onDestroy();
    _stopLineArrival_controller_xjal.onDestroy();
    _stopLineEntry_controller_xjal.onDestroy();
    super.onDestroy();
  }

  @AnyLogicInternalCodegenAPI
  @Override
  public void doFinish() {
    super.doFinish();
    trainSource.doFinish();
    trainMoveTo.doFinish();
    trainDecouple.doFinish();
    trainCouple.doFinish();
    trainDispose.doFinish();
    trainMoveTo1.doFinish();
    trainMoveTo2.doFinish();
    trainMoveTo3.doFinish();
    trainMoveTo4.doFinish();
    delay.doFinish();
    trainDecouple1.doFinish();
    delay1.doFinish();
    selectOutput.doFinish();
    trainMoveTo5.doFinish();
    trainMoveTo6.doFinish();
    trainCouple1.doFinish();
    selectOutput1.doFinish();
    trainCouple2.doFinish();
    trainMoveTo7.doFinish();
    trainDispose1.doFinish();
    trainMoveTo8.doFinish();
    newLoco.doFinish();
    _stopLineN8_controller_xjal.doFinish();
    _stopLineN7_controller_xjal.doFinish();
    _stopLineN5_controller_xjal.doFinish();
    _stopLineN4_controller_xjal.doFinish();
    _stopLineN3_controller_xjal.doFinish();
    _stopLineN6_controller_xjal.doFinish();
    _stopLineN1_controller_xjal.doFinish();
    _stopLineN2_controller_xjal.doFinish();
    _stopLineN11_controller_xjal.doFinish();
    _stopLineN12_controller_xjal.doFinish();
    _stopLineN9_controller_xjal.doFinish();
    _stopLineN10_controller_xjal.doFinish();
    _stopLineHump_controller_xjal.doFinish();
    _stopLineArrival_controller_xjal.doFinish();
    _stopLineEntry_controller_xjal.doFinish();
  }

  /**
   * 这个数字在这里是出于模型快照存储的目的。它不应该被用户修改。
   */
  @AnyLogicInternalCodegenAPI
  private static final long serialVersionUID = 2882341054946184571L;


}
